"""
-------------------------------------------------------------------------------
This is the demo LI data plotter for quickly checking the contents of
LI time and parallax corrected flash/group/event files created using
the main_li_data_reader.py.
-------------------------------------------------------------------------------
Copyright EUMETSAT, author(s) Sven-Erik Enno, 2025.
-------------------------------------------------------------------------------
"""

import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import xarray as xr
import numpy as np
import matplotlib as mpl
import pickle


class Area:
    def __init__(self, north=None, south=None, west=None, east=None, ):
        self.north = north
        self.south = south
        self.west = west
        self.east = east


def grid_totals(llats, llons, area=None, reso=1):
    """
    Computes the totals of lightning LGR/LFL lat/lon points per grid cell.
    Args:
      llats, llons = xarray data arrays of latitudes and longitudes.
      area = an area class object representing the area of interest.
      reso = grid resolution in degrees.
    Returns:
      totals = xarray data array containing the latitudes, longitudes and
        totals of lightning observations for each grid cell.
    """

    # The lat/lon bounds of the grid.
    if area is not None:
        latn, lats, lonw, lone = area.north, area.south, area.west, area.east
    else:
        latn, lats, lonw, lone = 85, -85, -85, 85

    # All lat and lon bins as 1d arrays.
    lat_bins = np.arange(lats, latn + reso, reso)
    lon_bins = np.arange(lonw, lone + reso, reso)

    totals = np.histogram2d(llons, llats, bins=(lon_bins, lat_bins), range=[(lonw, lone), (lats, latn)])[0]

    totals = xr.DataArray(totals.T, coords=[lat_bins[:-1], lon_bins[:-1]],
        dims=['latitude', 'longitude'])

    totals['latitude'] = totals['latitude'] + reso / 2
    totals['longitude'] = totals['longitude'] + reso / 2

    return totals


def plot_li_fov_polygons(ax, projection=ccrs.PlateCarree()):
    """
    Plot the polygons of LI camera field-of-views on a map.
    Args:
      ax: axis with cartopy map to plot the polygons on.
    Returns:
      None, plots the polygons on the map.
    """

    polygons = [{'name': 'west', 'color': 'dodgerblue', 'file_path': './li_fov/LI_west.p'},
                {'name': 'north', 'color': 'darkorange', 'file_path': './li_fov/LI_north.p'},
                {'name': 'east', 'color': 'green', 'file_path': './li_fov/LI_east.p'},
                {'name': 'south', 'color': 'darkred', 'file_path': './li_fov/LI_south.p'}]

    for polygon in polygons:
        file_path = polygon['file_path']
        with open(file_path, 'rb') as pickle_file:
            poly = pickle.load(pickle_file)
        color = polygon['color']
        ax.add_geometries(poly, crs=projection, facecolor='none', edgecolor=color, lw=2)

    return


def plot_accmap(acc_grid, projection=ccrs.PlateCarree(), gridlines=True,
    vmax=None, area=None, plot_fov=False, title='', fname=None):
    """
    Plot the accumulated lightning locations heatmap.
    Args:
      acc_grid = xarray ready-to-plot lightning accumulation array.
      projection = map projection.
      gridlines = if True then plot map gridlines.
      vmax  = maximum value of observations per grid cell for colour scale.
      area = list defing the bounding box of the area or interest like [north, south, west, east].
      plot_fov = if True then plot the 4 LI camera field-of-views on the map.
      title = plot title str.
      fname = output file name, if given then save plot as file.
    Returns:
      None, displays and/or saves the plot.
    """

    if area is None:
        area = Area(85, -85, -85, 85)
    else:
        area = Area(area[0], area[1], area[2], area[3])

    # Create the figure, plot background map and title.
    fig, ax = plt.subplots(subplot_kw=({'projection': projection}))
    format_fig_size(fig, area)
    ax = plotmap(ax, gridlines=gridlines, area=area)
    fig.suptitle(title, y=0.92)

    # Plot the heatmap.
    if not vmax:
        vmax=5*10**3
    acc_grid.plot(ax=ax, transform=ccrs.PlateCarree(),
        norm=mpl.colors.LogNorm(vmin=1, vmax=vmax), cmap='gist_rainbow_r')

    # Plot area ploygons if required.
    if plot_fov:
        plot_li_fov_polygons(ax, projection=projection)

    # Save if file name is given.
    if fname:
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved the plot as {fname}!')

    return


def plotmap(ax, set_global=False, gridlines=False, area=None):
    """
    Initializes the figure and plots cartopy map with coastlines, country
    borders and gridlines.
    Args:
      ax = matplotlib axis object to plot the map on.
      set_global = if True then set the extent of the ax to the limits of the
        projection (can be actually smaller than global if national projection).
      gridlines = if True then plot gridlines.
      area = an area class object representing the area of interest.
    Returns:
      ax = input matplotlib axis object with the map.
    """
 
    # Add main map features.
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
 
    # Set to the extent of the projection if required.
    if set_global:
        ax.set_global()
 
    # Plot gridlines if required.
    if gridlines:
        try:
            gridlines = ax.gridlines(draw_labels=True)
            gridlines.top_labels = None
            gridlines.right_labels = None
        except TypeError:
            gridlines = ax.gridlines()
 
    # Zoom into a sepecific area.
    if area:
        ax.set_xlim(area.west, area.east)
        ax.set_ylim(area.south, area.north)
 
    return ax
 
 
def format_fig_size(fig, area):
    """
    Spatial regions can be horizontal or vertical in shape. This function
    measures the sides of the lat/lon box in degrees and formats the figure
    size for best layout in the saved image (not so optimal for the live image
    window.
    Args:
      fig = matplotlib fig object to format.
      area = an area class object representing the area of interest.
    Returns:
      None, sets the figure width and height to the computed value, based
      on the shape of the area.
    """
 
    y_x_ratio = (area.north - area.south) / (area.east - area.west)
    # 8 and 6 are experimental and work best on saved plots.
    width = int(8 / y_x_ratio)
    height = 6
    fig.set_figwidth(width)
    fig.set_figheight(height)
 
 
def plot_lmap(lats, lons, time, projection=ccrs.PlateCarree(), gridlines=True,
              area=None, area_nswe=None, plot_fov=False, title=None):
    """
    Plot events/groups/flashes, coloured by hour of occurrence.
    Args:
      lats = xarray of latitudes
      lons = array of longitudes
      time = array of time information
      projection = map projection.
      gridlines = if True then plot map gridlines.
      area = an area class object defining the map extent in space.
      area_nswe = 4-element list of North South West East limits for area plot
      plot_fov = if True then plot the 4 LI camera field-of-views on the map.
      title = plot title str.
    Returns:
      None, displays and/or saves the plot.
    """

    if area_nswe is not None:
        area = Area(north=area_nswe[0], south=area_nswe[1], west=area_nswe[2], east=area_nswe[3])
    else:
        area = Area(85, -85, -85, 85)
 
    # Create the figure, plot background map and title.
    fig, ax = plt.subplots(subplot_kw=({'projection': projection}))
    format_fig_size(fig, area)
    ax = plotmap(ax, gridlines=gridlines, area=area)
    if title is not None:
        ax.set_title(title)
 
    # Create the colour bar.
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('gist_rainbow_r', 24))
    sm.set_clim(0, 24)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([h + 0.5 for h in range(24)])
    cbar.set_ticklabels([str(h).zfill(2) if h % 2 == 0 else "" for h in range(24)])
    cbar.set_label('Hour of Day')

    # Plot lightning observations.
    ax.scatter(lons, lats, transform=ccrs.PlateCarree(), cmap='gist_rainbow_r',
               marker='x', s=6, c=time.dt.hour, vmin=0, vmax=23)

    # Plot area polygons if required.
    if plot_fov:
        plot_li_fov_polygons(ax, projection=projection)
 
    return fig,ax


def print_dataset_demo(li_dataset):
    """
    Print a small LI xarray Dataset demo.
    Args:
      li_dataset: xarray Dataset as read from LI time and parallax corrected file(s).
    Returns:
      None, just prints some information and examples.
    """

    print('=' * 80)
    print('Printing the structure of the dataset where variable names can be seen:')
    print('=' * 80)
    print(li_dataset)
    print('=' * 80)
    print("Individual variables can be accessed as li_dataset['time'].values:")
    print('=' * 80)
    print(li_dataset['time'].values)
    print('=' * 80)
    print("It is also easy to convert the data to Pandas Dataframe using li_dataset.to_dataframe():")
    print('=' * 80)
    print(li_dataset.to_dataframe())
    print('=' * 80)
    print('''Data can be easily filtered. Here we filter for a spatial box of -20 to 20 degrees latitude and longitude:
    li_dataset = li_dataset.where(((li_dataset['latitude'] <= 20) & 
                                   (li_dataset['latitude'] >= -20) &
                                   (li_dataset['longitude'] >= -20) &
                                   (li_dataset['longitude'] <= 20)), drop=True)''')
    print('=' * 80)
    li_dataset = li_dataset.where(((li_dataset['latitude'] <= 20) &
                                   (li_dataset['latitude'] >= -20) &
                                   (li_dataset['longitude'] >= -20) &
                                   (li_dataset['longitude'] <= 20)), drop=True)
    print(li_dataset)
    print('=' * 80)
    print("End of the demo!")
    print('=' * 80)

    return


def load_corrected_data(file_paths):
    """
    Load LI time and parallax corrected data from daily/hourly/minutely file(s).
    Args:
      file_paths: str list of one or more file paths to load,
    Returns:
      li_dataset: xarray Dataset, time ordered LI observations in the input file(s).
    """

    li_dataset = xr.concat([xr.open_dataset(f) for f in file_paths], dim='time')
    li_dataset = li_dataset.sortby('time')
    print(f"Loaded {len(li_dataset['time'])} LI observations from the input file(s)!")

    return li_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Firstly, args common to all functions.
    common = argparse.ArgumentParser(description='''A small demo on how to use LI light travel time
        and parallax corrected data files.''')
    common.add_argument('output_type', help='''Choose output type: 'map' to plot all
        lightning locations on a map, coloured by hour; 'accmap' to plot a lightning accumulation
        map, 'print' to print the structure of the dataset with some demo operations.''',
        type=str, choices=['map', 'accmap', 'print'])
    common.add_argument('file_paths', nargs='+', help='''Full path of the file(s) to plot.''',
        type=str, default=[])
    common.add_argument('-li_fov', help='''Option to plot on the maps the field-of-view polygons
        of the LI 4 cameras.''', action='store_true')
    args = common.parse_args()

    # Load the data from time and parallax corrected LI data file(s).
    li_dataset = load_corrected_data(args.file_paths)

    # Plot or print the data.
    if args.output_type == 'map':
        times = li_dataset['time']
        latitudes = li_dataset['latitude']
        longitudes = li_dataset['longitude']
        plot_lmap(latitudes, longitudes, times, title='LI Lightning map', plot_fov=args.li_fov)
        plt.show()
    elif args.output_type == 'accmap':
        latitudes = li_dataset['latitude']
        longitudes = li_dataset['longitude']
        acc_grid = grid_totals(latitudes, longitudes)
        plot_accmap(acc_grid, title='LI Lightning accumulation map', plot_fov=args.li_fov)
        plt.show()
    else:
        print_dataset_demo(li_dataset)

