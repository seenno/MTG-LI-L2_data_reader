"""
-------------------------------------------------------------------------------
This is the main LI data reader for retrieving LIL2 data from MTG-LI data 
files. Automatically applies light-travel time and parallax correction.
-------------------------------------------------------------------------------
Copyright EUMETSAT, author(s) Sven-Erik Enno, 2025.
-------------------------------------------------------------------------------
"""
import sys
import os
from datetime import datetime, timedelta
import argparse
import xarray as xr
import numpy as np
import warnings
import pandas as pd
import satpy
from satpy import Scene, find_files_and_readers

# Custom modules
import config
from li_parallax_corrector import apply_time_parallax_correction

# Load configuration
cfg = config.load()



def str_to_datetime(datetime_str):
    """
    Convert the input datetime string to a datetime object.
    Args:
        datetime_str: str datetime as day (YYYYMMDD), hour (YYYYMMDDHH), minute (YYYYMMDDHHMM) or
          second (YYYYMMDDHHMMSS).
    Returns:
        dtime: a datetime object corresponding to the input string.
    """

    if len(datetime_str) == 8:
        dtime = datetime.strptime(datetime_str, '%Y%m%d')
    elif len(datetime_str) == 10:
        dtime = datetime.strptime(datetime_str, '%Y%m%d%H')
    elif len(datetime_str) == 12:
        dtime = datetime.strptime(datetime_str, '%Y%m%d%H%M')
    elif len(datetime_str) == 14:
        dtime = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
    else:
        raise ValueError('Unknown datetime string format!')

    return dtime


def period_start_end(period):
    """
    Convert the input datetime string to two datetime objects, defining the start and end times
    of the time period.
    Args:
        period: str single datetime as day (YYYYMMDD), hour (YYYYMMDDHH), minute (YYYYMMDDHHMM) or second
          (YYYYMMDDHHMMSS), or two datetime string separated by '..' like 202503041200..202503041300.
    Returns:
        start, end: datetime objects corresponding to the start and end times of the input period.
    """

    if '..' in period:
        start_str, end_str = period.split('..')
    else:
        start_str, end_str = [period for i in range(2)]

    start = str_to_datetime(start_str)
    end = str_to_datetime(end_str)

    # For single period string define end as the start of next day/hour/minute/second.
    if '..' not in period:
        if len(period) == 8:
            end += timedelta(days=1)
        elif len(period) == 10:
            end += timedelta(hours=1)
        elif len(period) == 12:
            end += timedelta(minutes=1)
        elif len(period) == 14:
            end += timedelta(seconds=1)
        else:
            raise ValueError('Unknown datetime string format!')

    return start, end


def find_li_files(in_dir, product, start, end):
    """
    Find LI data files in a given directory during a given period of time and return their full paths in a
    sorted list. Note that files fully or partially between the start and end time are both returned.
    Args:
      in_dir: str, full path of the input dir containing  the LI files,
      product: str, li product to look for, can be l1bbck, l1ble, l2le, l2lef, l2lgr or l2lfl,
      start: datetime start time of the period of interest,
      end: datetime end time of the period of interest,
    Returns:
      li_files: listr str, a sorted list of full paths of the LI files fully or partially in the period of interest.
    """

    reader = cfg['products'][product]['reader']

    li_files = find_files_and_readers(start_time=start, end_time=end, base_dir=in_dir, reader=reader)[reader]

    # Filter file paths for a specific L1B/L2 product and sort.
    fn_pattern = cfg['products'][product]['fn_pattern']
    li_files = sorted([f for f in li_files if fn_pattern in os.path.basename(f)])

    return li_files


def apply_li_l1b_filter(dataset):
    """
    Apply LIL1B filter. The raw LI-1B-LE files actually contain the whole L0 dataset with
    L1B filtering flags. This function uses the L1B filtering flag values and returns
    events that passed L1B filtering (all filtering values 0). This is the L1B dataset.
    Args:
      dataset: xarray LIL0 events dataset to filter.
    Returns:
      dataset: xarray L1B events dataset.
    """

    # Apply all L1B Filters
    condition = (
            (dataset['preprocessing_filter_probability'] == 0) &
            (dataset['rts_noise_filter_probability'] == 0) &
            (dataset['ghost_noise_filter_probability'] == 0) &
            (dataset['hybrid_filter_probability'] == 0) &
            (dataset['jitter_noise_filter_probability'] == 0) &
            (dataset['stc_filter_probability'] == 0)).compute()

    dataset = dataset.where(condition, drop=True)

    return dataset


def remove_redundant_event_vars(dataset, ds_name):
    """
    Event files contain variables that are used for computing event times. This
    filter removes these vars after event times have been computed to keep the
    output dataset smaller and assuming that most users won't care about these
    vars as they get event times anyway.
    Args:
      dataset: xarray LIL0/L1B/L2 events dataset to filter,
      ds_name: product name as it appears in the input,
    Returns:
      dataset: xarray LIL0/L1B/L2 events dataset with the redundant vars removed.
    """

    if ds_name in ['l0le', 'l1ble']:
        redundant_vars = ['event_chunk_id',
                          'integration_frame_index',
                          'integration_start_time',
                          'local_frame_index']
    elif ds_name in ['l2lef']:
        redundant_vars = ['epoch_time',
                          'time_offset']
    else:
        redundant_vars = []

    dataset = dataset.drop_vars(redundant_vars)

    return dataset


def remove_l1b_filter_vars(dataset):
    """
    Remove L1B filter variables from L0/L1B dataset. Should be the default behaviour
    for L1B data as all events in L1B dataset have all L1B filter values equal to 0.
    Args:
      dataset: xarray LIL0/L1B events dataset to filter.
    Returns:
      dataset: xarray LIL0/L1B events dataset with the 6 L1B filter vars removed.
    """

    dataset = dataset.drop_vars([var for var in dataset.data_vars if 'filter' in var])

    return dataset


def remove_event_radiance_windows(dataset):
    """
    For each event in the LI-1B-LE files, the event radiance ('pixel_radiance' var) and its
    background radiance ('pixel_background_radiance' var) come as 3x3 pixel windows where
    the central pixel is the actual event/background radiance while the other 8 pixels are
    the radiance of the pixels surrounding the event pixel.
    Normally users only want the event (background) radiance. This function removes the
    3x3 radiance windows by only keeping the event (i.e. central pixel) radiance and its
    background radiance. The window_row and window_column dimensions are also removed
    as they are no longer needed.
    Args:
      dataset: xarray LIL0/L1B events dataset to filter.
    Returns:
      dataset: xarray LIL0/L1B events dataset with the 6 L1B filter vars removed.
    """

    # Select the central pixel (index 1,1) from the 3x3 window
    dataset["pixel_background_radiance"] = dataset["pixel_background_radiance"].isel(window_row=1, window_column=1)
    dataset["pixel_radiance"] = dataset["pixel_radiance"].isel(window_row=1, window_column=1)

    return dataset


def filter_rc_box(dataset, rc_box):
    """
    Filter LI event xarray dataset for a specific LI camera pixel grid row/column box.
    Args:
      dataset: xarray dataset LI events,
      rc_box: list int, len=4, defines the row/column box to filter the events for, shall
         be given as [row_min, row_max, col_min, col_max], e.g. rc_box=[0, 199, 0, 299] to
         get events in the first 200 rows and 300 columns of the pixel grid,
    Returns:
      dataset: xarray dataset, input filtered for the row/column box.
    """

    row_min, row_max, col_min, col_max = rc_box
    dataset = dataset.where(((dataset['central_pixel_detector_column'] <= col_max) &
                             (dataset['central_pixel_detector_column'] >= col_min) &
                             (dataset['central_pixel_detector_row'] >= row_min) &
                             (dataset['central_pixel_detector_row'] <= row_max)), drop=True)

    return dataset


def filter_ll_box(dataset, ll_box):
    """
    Filter LI event/groups/flashes xarray dataset for a specific lat/lon box.
    Args:
      dataset: xarray dataset LI events,
      ll_box: list float, return LI events/groups/flashes only within a latitude/longitude box,
        shall be given as [lat_north, lat_south, lon_west, lon_east], e.g. ll_box=[10, -5,-10, 15]
        returns events/groups/flashes between 10 deg N to 5 deg S and 10 deg W to 15 deg E,
    Returns:
      dataset: xarray dataset, input filtered for the latitude/longitude box.
    """

    lat_var_name = [var for var in dataset.variables if "latitude" in var][0]
    lon_var_name = [var for var in dataset.variables if "longitude" in var][0]
    lat_max, lat_min, lon_min, lon_max = ll_box
    dataset = dataset.where(((dataset[lat_var_name] <= lat_max) &
                             (dataset[lat_var_name] >= lat_min) &
                             (dataset[lon_var_name] >= lon_min) &
                             (dataset[lon_var_name] <= lon_max)), drop=True)

    return dataset


def create_xr_dataset(arrays, ds_name, sector=None):
    """
    Create xarray dataset for L2LFL or L2LGR file, or for a sector in the
    LIL1B LE, L2LE or L2LEF file. Computes the times of all observations
    if not provided in the input data file. Also adds sector as an
    additional variable, if event data.
    Args:
      arrays: dict of {name: xr data array} pairs, each representing a var
        load from the input Satpy scene,
      ds_name: product name as it appears in the input,
      sector: str 'west', 'north', 'east' or 'south' if event data,
    Returns:
      dataset: xarray dataset containing the values of all vars in the
        input file plus computed event time and added sector vars, if event
        dataset.
    """

    # If a specific sector then remove arrays of other sectors.
    if sector:
        arrays = {k: v for k, v in arrays.items() if sector in k}

    # Clear attributes, otherwise netCDF writer will complain about time index 
    # format (introduced by Taylan, details unknown to RSP/SEE).
    for v in arrays.values():
        v.attrs.clear()

    if sector is not None:
        print(f'Working with sector {sector}...')
    # If events then compute their times as these are not in the LI input files.
    if ds_name in ['l0le', 'l1ble', 'l2lef']:
        if ds_name in ['l0le', 'l1ble']:
            # Add 1 frame to get LE times as their frame end times in the computation below.
            integration_frame_index = arrays[f'integration_frame_index_{sector}_sector'].astype(int) + 1
            integration_start_times = arrays[f'integration_start_time_{sector}_sector']
            etime = integration_start_times + (integration_frame_index * 1e6).astype('timedelta64[ns]')
        elif ds_name in ['l2lef']:
            etime = arrays[f'epoch_time_{sector}_sector'] + arrays[f'time_offset_{sector}_sector']
        time_var_name = f'time'
        arrays[time_var_name] = etime
        n_lightnings = len(etime)
    else:
        n_lightnings = len(arrays['longitude'])

    dataset = xr.Dataset()
    for array_name, array in arrays.items():
        # We are not interested in vars whose length =! n of events (e.g. frame data).
        if len(array) != n_lightnings:
            continue
        # Remove sector information from var names.
        if sector:
            array_name = array_name.replace(f'_{sector}_sector', '')
        dataset.update({array_name: array})

    dataset = dataset.set_index({'y': 'time'})
    dataset = dataset.rename({'y': 'time'})

    if 'crs' in list(dataset.coords):
        dataset = dataset.drop_vars('crs')

    # Add sector and OC flags as vars if the data comes per sector.
    if sector:
        dataset[f'sector'] = ([f'time'], [sector for i in range(n_lightnings)])

    return dataset


def load_data_to_xr(data_dir, ds_name, start_time, end_time, sectors=None, rad_win=False,
                    l1b_filt=False, rc_box=None, ll_box=None):
    """
    Load parallax and light travel time corrected LIL0, LIL1B or L2 event/group/flash dataset
    from raw LI file(s) to time-indexed xarray dataset.
    Args:
      data_dir: str full path of the input directory containing the LI L1B/L2 files,
      ds_name: str name of a LIL1B/L2 dataset to load, one of the following:
          l0le: LI L0 lightning events, i.e. all events that LI sent to the ground,
          l1ble: LI L1B lightning events, i.e. all events that passed L1B filtering,
          l2lef: LI L2 lightning vents in flashes, i.e. all lightning events that passed the L2
                 filtering and are used in groups/flashes.
          l2lgr: LI L2 lightning groups,
          l2lfl: LI L2 lightning flashes,
      start_time: datetime start time of the period of interest (included),
      end_time: datetime end time of the period of interest (excluded),
      sectors: str list (events only), load only events in particular LI sector(s), (i.e. observed
        by specific LI camera(s), if not specified then all events in all sectors are imported,
        equals to sectors=['west', 'north', 'east', 'south'],
      rad_win: bool (L0/L1B events only), if True keep the 3x3 pixel event radiance and event
        background radiance windows (event as the central pixel + its 8 neighbour pixels). By
        default, only the central pixel information is returned as this is the event (background)
        radiance that most users are interested in,
      l1b_filt: bool (L0/L1B events only), if True return also L1B filtering results, these are 6
        filtering values per L0/L1B event that determine which events passed the L1B filtering step,
        by default they are not shown in the output dataset,
      rc_box: list int (events only), filter LI events dataset for a specific LI camera pixel grid
        row/column box, shall be given as [row_min, row_max, col_min, col_max], e.g. use
        sectors=['east'] and rc_box=[0, 199, 0, 299] to get events in the first 200 rows and 300
        columns of the east camera pixel grid,
      ll_box: list float, return LI events/groups/flashes only within a latitude/longitude box,
        shall be given as [lat_north, lat_south, lon_west, lon_east], e.g. ll_box=[10, -5,-10, 15]
        returns events/groups/flashes between 10 deg N to 5 deg S and 10 deg W to 15 deg E,
    Returns:
      dataset: all loaded var(s) from all loaded LI file(s) that meet the user-selected space-time
        filtering criteria as a time-indexed xarray dataset.
    """

    # First, find the relevant LI data files.
    print('=' * 80)
    print(f'Looking for LI {ds_name} files...')
    print('-' * 80)
    li_files = find_li_files(data_dir, ds_name, start_time, end_time)
    print(f'Found {len(li_files)} {ds_name} file(s) in {data_dir} during {start_time} to {end_time}.')
    if len(li_files) < 1:
        raise FileNotFoundError(f'No {ds_name} files found in {data_dir}!')

    # Using relevant files, create Satpy scene and load necessary vars to memory.
    print('=' * 80)
    print(f'Reading the data from the file(s)...')
    reader = cfg['products'][ds_name]['reader']
    scn = satpy.Scene(reader=reader, filenames=li_files)
    available_dataset_names = scn.available_dataset_names()
    scn.load(wishlist=available_dataset_names)
    arrays = {k['name']: scn[k['name']] for k in scn.keys()}

    # Make sure that time always appears in 'time' field for LGR/LFL data.
    if ds_name in ['l2lgr', 'l2lfl']:
        old_time_key = [k for k in available_dataset_names if 'time' in k][0]
        arrays['time'] = arrays.pop(old_time_key)

    # Events come as per sector/OC while groups/flashes come for the whole LI FOV.
    if ds_name in ['l2lgr', 'l2lfl']:
        dataset = create_xr_dataset(arrays, ds_name, sector=None)
    else:
        print('-' * 80)
        sector_datasets = []
        if sectors is not None:
            sectors = sectors
        else:
            sectors = ['west', 'north', 'east', 'south']
        for sector in sectors:
            dataset = create_xr_dataset(arrays, ds_name, sector=sector)
            sector_datasets.append(dataset)
        dataset = xr.concat(sector_datasets, 'time')

    # Apply LI L1B filter, if L1B dataset is needed.
    if ds_name == 'l1ble':
        dataset = apply_li_l1b_filter(dataset)

    # Remove or keep more specific vars, not needed by every user.
    if ds_name in ['l0le', 'l1ble']:
        if not l1b_filt:
            dataset = remove_l1b_filter_vars(dataset)
        if not rad_win:
            dataset = remove_event_radiance_windows(dataset)
    if ds_name in ['l0le', 'l1ble', 'l2lef']:
        dataset = remove_redundant_event_vars(dataset, ds_name)

    # Apply parallax and light-travel-time correction.
    print('=' * 80)
    print('Applying parallax and light travel time correction...')
    print('-' * 80)
    apply_time_parallax_correction(dataset, start_time)

    # To overcome 'lazy' load and really retrieve the data, pre-required for time slicing below.
    dataset = dataset.compute()

    print('=' * 80)
    print('Filtering for time and space, if needed...')

    # Finally filter for the input time range as the input files normally cover somewhat longer time range.
    dataset = dataset.sortby(f'time')
    mask = (dataset.time >= np.datetime64(start_time)) & (dataset.time < np.datetime64(end_time))
    dataset = dataset.sel(time=mask)

    # Select events in a row/col box.
    if rc_box is not None:
        dataset = filter_rc_box(dataset, rc_box)

    # Select events/groups/flashes in a row/col box.
    if ll_box is not None:
        dataset = filter_ll_box(dataset, ll_box)
    print('=' * 80)

    return dataset


def save_xarray_dataset(dataset, period, ds_name, out_dir):
    """
    Save the parallax and light travel time corrected xarray dataset from load_data_to_xr()
    in a minutely, hourly or daily *.nc file for quick access in the future.
    Args:
      dataset: xarray dataset, parallax and light travel time corrected LI events, groups or
        flashes returned by load_data_to_xr(),
      period: str data period from user input, can be full day (YYYYMMDD), full hour (YYYYMMDDHH)
        or full minute (YYYYMMDDHHMM) to use this saving functionality,
      ds_name: str name of a LIL1B/L2 dataset to load, lil0le, l1ble, l2lef, l2lgr or l2lfl,
      out_dir: str full path of the output directory for saving the file,
    Returns:
      None, saves the dataset in a *.nc file.
    """

    print('Saving the dataset in a *.nc file...')
    print('-' * 80)

    # Construct output file path.
    fn_pattern = cfg['products'][ds_name]['fn_pattern']
    if ds_name == 'lil0le':
        fn_pattern = fn_pattern.replace('1', '0')
    fname = f"{fn_pattern}{period}_corrected.nc"
    out_file_path = os.path.join(out_dir, fname)
    print(out_file_path)

    # Write the data into the output file.
    # Keep the time dimension to preserve the time-indexed xarray dataset structure.
    dims_to_remove = [d for d in dataset.dims if d != 'time']
    dataset = dataset.drop_dims(dims_to_remove)
    # Apply also some compression to reduce the file size but keep fast access.
    comp = dict(zlib=True, complevel=4)
    #encoding = {var: comp for var in dataset.data_vars}
    encoding = {var: comp for var in dataset.data_vars if dataset[var].dtype.kind in 'fi'}
    dataset.to_netcdf(path=out_file_path, encoding=encoding)
    print(f'Saved parallax and light travel time corrected data in {out_file_path}.')

    return



def Main():
    # Firstly, args common to all functions. 
    common = argparse.ArgumentParser(description='''Caller for reading and processing LI-L2-LEF,
        LI-L2-LGR or LI-L2_LFL datasets.''')
    common.add_argument('period', help='''Period to import in UTC, can be a day like '20250304', an hour 
        like '2025030410', a minute like 202503041033, a second like 20250304103308 or start (included) and end
        (excluded) time of a period like 202503041255..202403041305''', type=str)
    common.add_argument('--in_dir', help='''Source directory of the raw LIL1B/L2 files, default 
        ./downloaded_data.''', type=str, default='./downloaded_data')
    common.add_argument('--out_dir', help='''Target directory for saving the processed LIL1B/L2 data
        in nc files, default ./processed_data.''', type=str, default='./processed_data')
    common.add_argument('-ll_box', nargs='+', help='''Return only LI observations only within this 
        latitude/longitude box, shall be given as [lat_north, lat_south, lon_west, lon_east], e.g. [10, -5,-10, 15] 
        filters for a box of 10 deg N to 5 deg S and 10 deg W to 15 deg E.''', default=None, type=float)
    common.add_argument('-s', '--save', help='''Save the parallax and light travel time corrected dataset
        in a *.nc file. The period must be full day (YYYYMMDD), full hour (YYYYMMDDHH) or full minute (YYYYMMDDHHMM)''',
        action='store_true')
    events = argparse.ArgumentParser(add_help=False)
    events.add_argument('-sec', '--sector', nargs='+', help='''Retrieve events for a specific sector(s) 
        only, choose from west, north, south, east.''', type=str, choices=['west', 'north', 'east', 'south'],
        default=None)
    events.add_argument('-rc_box', nargs='+', help='''Return only events within a row/column box in the LI
        camera pixel grid (choose the camera using --sector), e.g. 0 200 0 300 returns events in the first 200 rows and 
        300 columns.''', default=None, type=int)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    # Additional arguments for the reprocessing mode.
    l2lef = subparsers.add_parser('l2lef', parents=[common, events], help='''Get LI-L2-LEF data (LI events
        used in flashes).''', add_help=False)
    l2lgr = subparsers.add_parser('l2lgr', parents=[common], help='''Get LI-L2-LGR data (LI groups).''',
        add_help=False)
    l2lfl = subparsers.add_parser('l2lfl', parents=[common], help='''Get LI-L2-LFL data (LI flashes).''',
        add_help=False)
    args = parser.parse_args()

    if args.save:
        if '..' in args.period or len(args.period) < 8 or len(args.period) > 12:
            raise ValueError('Saving only works with a period in full days, hours or minutes!')

    sectors = None
    if 'sector' in args:
        sectors = args.sector

    rc_box = None
    if 'rc_box' in args:
        rc_box = args.rc_box

    start, end = period_start_end(args.period)

    # Call the upper level raw data processing wrapper.
    print(f'LI data reader started at {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.')
    print(f"Retrieving and processing {cfg['products'][args.func]['fn_pattern']} during {args.period}.")

    dataset = load_data_to_xr(args.in_dir, args.func, start, end, sectors=sectors, rc_box=rc_box, ll_box=args.ll_box)

    print(f"Found {len(dataset['time'])} LI observations from {dataset['time'].min().values} to {dataset['time'].max().values}")
    if args.save:
        save_xarray_dataset(dataset, args.period, args.func, args.out_dir)
    else:
        print('Printing the dataset structure just for information. Use -s/--save to save it for later use.')
        print('=' * 80)
        print(dataset)


if __name__ == '__main__':
    Main()
