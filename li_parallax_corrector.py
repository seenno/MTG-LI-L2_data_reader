"""
-------------------------------------------------------------------------------
This is the LI parallax and LTT correction code.
-------------------------------------------------------------------------------
Copyright EUMETSAT, author(s) Alessio Bozzo, Sven-Erik Enno 2022-2023.
-------------------------------------------------------------------------------
"""

import warnings
import numpy as np
import xarray as xr
import dask.array as da
from pyresample import create_area_def
from pyresample.geometry import SwathDefinition
from pyresample.bucket import BucketResampler
import datetime
import time
import glob

# Custom modules
import config

# Load configuration
cfg = config.load()


def linear_time_interp_lut(day, month, year, lut_file, lut_dir):
    """
    Simple linear interpolation between months. The look-up tables
    are assumed representative of the 15th of the month.
    Inputs:
      day: day of the coordinates to process
      month: month of the coordinates to process
      year: year of the coordinates to process
      lut_file: full path of the LUT file to open
      lut_dir: full path of the directory storing the LUTs
    Output:
      px_lut: the xarray Dataset with the interpolated LUT
    """

    DoY = datetime.datetime(year, month, day, 0, 0, 0).timetuple().tm_yday
    if day == 15:
        px_lut = xr.open_dataset(lut_file, decode_times=False)
    else:
        print("Luts interpolated between months")
        if day < 15:
            # if January the preceding month will be December
            if month == 1:
                month_m1 = 12
            else:
                month_m1 = month - 1
            month_name_m1 = datetime.datetime.strptime(str(month_m1), "%m").strftime("%b")
            print(month_name_m1 + " and " + datetime.datetime.strptime(str(month), "%m").strftime("%b"))
            lut_file_m1 = glob.glob(lut_dir + 'PX_' + month_name_m1 + '_2005_2018_MK_res1*')[0]
            px_lut1 = xr.open_dataset(lut_file_m1, decode_times=False)
            px_lut2 = xr.open_dataset(lut_file, decode_times=False)
            DoY1 = datetime.datetime(year, month_m1, 15, 0, 0, 0).timetuple().tm_yday
            DoY2 = datetime.datetime(year, month, 15, 0, 0, 0).timetuple().tm_yday
        elif day > 15:
            # if December the following month will be January
            if month == 12:
                month_p1 = 1
            else:
                month_p1 = month + 1
            month_name_p1 = datetime.datetime.strptime(str(month_p1), "%m").strftime("%b")
            print(datetime.datetime.strptime(str(month), "%m").strftime("%b") + " and " + month_name_p1)
            lut_file_p1 = glob.glob(lut_dir + 'PX_' + month_name_p1 + '_2005_2018_MK_res1*')[0]
            px_lut1 = xr.open_dataset(lut_file, decode_times=False)
            px_lut2 = xr.open_dataset(lut_file_p1, decode_times=False)
            DoY1 = datetime.datetime(year, month, 15, 0, 0, 0).timetuple().tm_yday
            DoY2 = datetime.datetime(year, month_p1, 15, 0, 0, 0).timetuple().tm_yday

        px_lut = (px_lut2 - px_lut1) / (DoY2 - DoY1) * (DoY - DoY1) + px_lut1
        px_lut1.close()
        px_lut2.close()

    return px_lut


def resample_lut(data, lons_in, lats_in, gridres):
    """
    Regridding from irregular to regular grid using Bucket resampling method
    NOT USED AT THE MOMENT
    """

    CHUNK_SIZE = 'auto'
    target_def = create_area_def(area_id='my_area',
                                 projection={'proj': 'latlong'},
                                 center=(0.0, 0.0),
                                 width=180. / gridres,
                                 height=180. / gridres,
                                 resolution=gridres
                                 )
    lons = da.from_array(lons_in, chunks=CHUNK_SIZE)
    lats = da.from_array(lats_in, chunks=CHUNK_SIZE)

    resampler = BucketResampler(source_lats=lats, source_lons=lons, target_area=target_def)
    result = resampler.get_average(da.from_array(data, chunks=CHUNK_SIZE))

    return result


def resample_lut_nearest(data, lons, lats, gridres):
    """
    This function regrids from irregular to regular grid
    using nearest neighbor search.
    The area is predefined as the hemisphere with lats -90/90
    and lons -180/180
    Inputs:
      data: the data to regrid (numpy array)
      lons: the longitudes of the data (2d numpy array)
      lons: the latitudes of the data (2d numpy array)
      gridres: the resolution of the output regular grid in degrees
    Output
      result: the regridded data (numpy array)
    """

    from pyresample import kd_tree

    target_def = create_area_def(area_id='my_area',
                                 projection={'proj': 'latlong'},
                                 center=(0.0, 0.0),
                                 width=180. / gridres,
                                 height=180. / gridres,
                                 resolution=gridres
                                 )
    swath_def = SwathDefinition(lons=lons, lats=lats)
    result = kd_tree.resample_nearest(swath_def, data,
                                      target_def, radius_of_influence=50000, epsilon=0.5)

    return result


def initialise_lut(indate, lut_dir):
    """
    This function organises the initialisation of the LUT
    For the given date of the data to process, two actions are performed
    - the linear temporal interpolation to the input date
    - the regridding to a regular lat/lon grid from the original SEVIRI projection
    Inputs:
      indate: the date of the coordinates to process (datetime object)
      lut_dir: full path to the location where the LUTs are stored

    Output:
      px_latlon: the xarray dataset with the time interpolated and regridded LUT
               containing the parallax shift in degrees and the photon travel time
    """

    print("LUTs initialisation")
    date = datetime.datetime.strptime(indate, '%Y-%m-%d')
    month_name = datetime.datetime.strptime(str(date.month), "%m").strftime("%b")
    gridres = .125  # resolution of the regular lat lon grid to use for LUTs

    # read LUT. LUTs are representative of the 15th of the month.
    # linear interpolation between
    lut_file = glob.glob(lut_dir + 'PX_' + month_name + '_2005_2018_MK_res1*')[0]

    print("Time interpolation of LUT to date: " + str(date.day) + " " + month_name + " " + str(date.year))
    px_lut = linear_time_interp_lut(date.day, date.month, date.year, lut_file, lut_dir)

    print("Resampling LUTs to regular lat/lon gird at " + str(gridres) + "deg resolution")

    with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "You will likely lose important projection information",
                UserWarning,
            )
            pxDlon_resampled = resample_lut_nearest(px_lut['new_lon_ice'].values - px_lut['lon'].values,
                                                    px_lut['lon'].values, px_lut['lat'].values, gridres)
            pxDlat_resampled = resample_lut_nearest(px_lut['new_lat_ice'].values - px_lut['lat'].values,
                                                    px_lut['lon'].values, px_lut['lat'].values, gridres)
            pxDtime_resampled = resample_lut_nearest(px_lut['temporal_shift'].values,
                                                     px_lut['lon'].values, px_lut['lat'].values, gridres)

    # create an xarray dataset with the resampled values to facilitate the look-up process
    px_latlon = xr.Dataset(
        data_vars=dict(
            pxDlon=(["lat", "lon"], pxDlon_resampled, {"units": "deg E"}),
            pxDlat=(["lat", "lon"], pxDlat_resampled, {"units": "deg N"}),
            pxDtime=(["lat", "lon"], pxDtime_resampled, {"units": "ms"}),
        ),
        coords=dict(
            lon=(["lon"], np.arange(-90., 90., gridres)),
            lat=(["lat"], np.arange(90., -90., -gridres)),
        ),
        attrs=dict(
            description="resampled lut",
        ),
    )

    print("end LUT initialisation")
    return px_latlon


def compute_new_latlon(lons, lats, px_latlon, date, lut_dir):
    """
    The function performs the search of the values in the LUT
    using the nearest neighbor method within the xarray Dataset structure.
    The values are then used to compute the new parallax-corrected coordinates.
    Inputs:
      lons: xarray DataArray with input longitudes
      lats: xarray DataArray with input latitudes
      px_latlon: the xarray DataSet with the LUT on a regular lat/lon grid
      date: datetime object with the date of the input coordinates to process
      lut_dir: full path of the location where the LUT are stored
    Output:
      px_nearest: xarray DataSet with the parallax-corrected coordinates, the
                photon-travel time and the original coordinates. Global attributes
                add extra metadata information.
    """

    total_start_time = time.time()
    px_nearest = px_latlon.sel(lon=lons, lat=lats, method='nearest')
    print("Input points processing done")
    print("--- total run time (s): %s ---" % (time.time() - total_start_time))

    print("computing new coords")
    new_lon = lons + px_nearest['pxDlon']
    new_lat = lats + px_nearest['pxDlat']
    print("--- total run time (s): %s ---" % (time.time() - total_start_time))

    px_nearest = px_nearest.assign(new_lon=new_lon)
    px_nearest = px_nearest.assign(new_lat=new_lat)
    # px_nearest['pxDtime'] = px_nearest['pxDtime']*1.e-3
    px_nearest['new_lon'].attrs = {'long_name': 'parallax shifted longitude', 'units': 'deg E'}
    px_nearest['new_lat'].attrs = {'long_name': 'parallax shifted latitude', 'units': 'deg N'}
    px_nearest['pxDlon'].attrs['long_name'] = 'longitude parallax shift'
    px_nearest['pxDlat'].attrs['long_name'] = 'latitude parallax shift'
    px_nearest['pxDtime'].attrs = {'long_name': 'photon travel time pixel to satellite', 'units': 'ms'}
    px_nearest.attrs = {'description': 'parallax corrected coordinates and photon travel time', \
                        'parallax LUT used': lut_dir, \
                        'processing date': date, \
                        'creation date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    return px_nearest


def apply_time_parallax_correction(dataset, start_time):
    """
    For a given LI L1B or L2 event/group/flash dataset, apply photon travel time
    correction for lightning times and parallax correction for lats/lons.
    Args:
        dataset: xarray Dataset of LI L1B/L2 events/groups/flashes.
        start_time: datetime start time of the period of interest.
    Returns:
        None, applies the correction to time/lat/lon fields of the input dataset.
    """

    # Choose the correct LUTs and interpolate correction for the input date.
    lut_dir = cfg['parallax_lut_dir']
    px_latlon = initialise_lut(start_time.strftime('%Y-%m-%d'), lut_dir)

    # Compute the LTT and parallax correction based on DT locations.
    lat_var_name = [var for var in dataset.data_vars if 'latitude' in var][0]
    lon_var_name = [var for var in dataset.data_vars if 'longitude' in var][0]
    dt_lat, dt_lon = dataset[lat_var_name], dataset[lon_var_name]
    correction = compute_new_latlon(dt_lon, dt_lat, px_latlon, start_time, lut_dir)

    # Now write the new corrected times, lats and lons to the input dataset.
    dataset[lat_var_name] = ('time', correction['new_lat'].values)
    dataset[lon_var_name] = ('time', correction['new_lon'].values)
    dataset['time'] = dataset['time'] - (correction['pxDtime'].values * 1e6).astype('timedelta64[ns]')