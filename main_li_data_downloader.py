"""
-------------------------------------------------------------------------------
This is the main LI data downloader for retrieving LI files form EUMETSAT
Data store. Contains some convenience features, including:
  - Automated extraction of the downloaded zip files.
  - Automated deletion of all the auxiliary files in the unzipped data files,
    only the real data files are kept.
  - Auto-check of the local data directory before any download to avoid
    re-downloading any data that already exists locally. This allows one to
    download the same dataset multiple times per day and every time only
    get the new files that were not previously available. 
-------------------------------------------------------------------------------
Copyright EUMETSAT, author(s) Sven-Erik Enno, 2025.
-------------------------------------------------------------------------------
"""

import argparse
import eumdac
from datetime import datetime
import shutil
import requests
import os
import glob
import re
import zipfile

# Custom modules
import config
from main_li_data_reader import period_start_end

# Load configuration
cfg = config.load()


def delete_auxiliary_files(download_dir):
    """
    Each LI product comes from the EUM Data Store with multiple auxiliary files (trailers,
    quicklooks, manifests) that are normally not needed by users. This script deletes all
    such files and only keeps the *BODY* files where the actual event/group/flash data is
    stored.
    Args:
      download_dir: str full path of the output directory where the downloaded and unzipped
        LI data files are located,
    Returns:
      None, removes unzipped LI product subdirectories and auxiliary files and keeps only
        the *BODY* files where LI event/group/flash data is stored.
    """

    # Get all subdirectories in the main directory
    subdirs = [d for d in sorted(os.listdir(download_dir)) if os.path.isdir(os.path.join(download_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(download_dir, subdir)
        # Find the BODY files that contain the LI event/group/flash data.
        pattern = os.path.join(subdir_path, '*BODY*')
        files_to_move = glob.glob(pattern)

        for file_path in files_to_move:
            new_path = os.path.join(download_dir, os.path.basename(file_path))
            shutil.move(file_path, new_path)

        # Remove the entire subdirectory
        shutil.rmtree(subdir_path)
        print(f'Deleted auxiliary files in {subdir_path}')

    return


def corresponding_nc_exists(zip_file, local_dir):
    """
    Check if a netCDF file corresponding to the remote zip already exists locally.
    Works for arbitrary product types (LI-2, LI-1B, etc.).
    Args:
      zip_file: str name of the zip file in the EUM Data Store,
      local_dir: str full path of the local file directory where the downloaded and unzipped
        data files are stored.
    Returns:
      nc_exists: bool, True if the corresponding netCDF file(s) already exist locally. 
    """
    
    nc_exists = False
    
    base = os.path.basename(zip_file)

    # Find all 14-digit timestamps in the file name (time created, start, and end).
    timestamps = re.findall(r'\d{14}', base)
    if len(timestamps) < 2:
        raise ValueError("Filename must contain at least two timestamps")

    # Read date from the second timestamp, i.e. time of the data start in the file.
    date_str = timestamps[1][:8]

    # Extract the RC counter.
    counter_match = re.search(r'_(\d{4})_0000$', base)
    if not counter_match:
        raise ValueError("Filename must end with _####_0000")
    counter_str = counter_match.group(1)

    # Take everything up to the first double dash (“--”), inclusive.
    prefix_match = re.search(r'--', base)
    if not prefix_match:
        raise ValueError("Filename does not contain '--' to anchor prefix")
    prefix = base[:prefix_match.end()]

    # Build the final glob nc file pattern and add output directory path. 
    base = f"{prefix}*{date_str}*{counter_str}_????.nc"
    pattern = os.path.join(local_dir, base)

    # Check if any existing nc file matches the pattern.
    if len(glob.glob(pattern)) > 0:
        nc_exists = True
    
    return nc_exists


def download_eum_data(ds_name, start_time, end_time, out_dir):
    """
    Download LI-L1B-LE, LI-L2-LEF, LI-L2-LGR or LI-L2_LFL data files from EUMETSAT Data Store,
    extract them and remove unnecessary auxiliary files and subdirectories.
    Args:
      ds_name: str dataset to download: l2lfl for L2 flashes, l2lgr for L2 groups, l2lef for L2
        lightning events in flashes, l0l1ble for L0/L1B evens (only available upon special request,
      start_time: datetime start time of the period of interest,
      end_time: datetime end time of the period of interest,
      out_dir: str full path of the output directory where the downloaded files are saved,
    Returns:
        dtime: a datetime object corresponding to the input string.
    """

    # Set the EUM Data Store access.
    print('=' * 80)
    print('Setting up EUM Data Store connection...')
    print('-' * 80)
    consumer_key = cfg['eum_dac']['consumer_key']
    consumer_secret = cfg['eum_dac']['consumer_secret']
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    try:
        print(f"This token '{token}' expires {token.expiration}")
    except requests.exceptions.HTTPError as error:
        print(f"Unexpected error: {error}")

    # Connect the EUM Data Store and look for the specific product.
    print('=' * 80)
    print('Looking for the requested data...')
    print('-' * 80)
    dataset_key = cfg['products'][ds_name]['eum_dac_key']
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(dataset_key)
    products = selected_collection.search(dtstart=start_time, dtend=end_time, sort='start,time,1')
    n_products = products.total_results
    print(f"{n_products} products found:")

    # Download the files from EUM Data Store and save in the output directory.
    print('=' * 80)
    print('Downloading the products...')
    print('-' * 80)
    n_downloads = 0
    for i, product in enumerate(products):
        if corresponding_nc_exists(str(product), out_dir):
            print(f"Skipping ({i+1}/{n_products}): {str(product)} already downloaded.")
            continue
        try:
             with product.open() as src_file, open(os.path.join(out_dir, src_file.name), mode='wb') as dest_file:
                shutil.copyfileobj(src_file, dest_file)
                print(f'Download of product ({i+1}/{n_products}) {product} finished.')
        except eumdac.product.ProductError as error:
            print(f"Error related to the product '{product}' while trying to download it: '{error.msg}'")
        except requests.exceptions.ConnectionError as error:
            print(f"Error related to the connection: '{error.msg}'")
        except requests.exceptions.RequestException as error:
            print(f"Unexpected error: {error}")
        n_downloads += 1

    # Extract the downloaded files as they come in ZIP format.
    if n_downloads > 0:
        print('=' * 80)
        print('Extracting the downloaded files...')
        print('-' * 80)
        for file in sorted(glob.glob(os.path.join(out_dir, '*.zip'))):
            if os.path.exists(file):
                try:
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        # Extract the content of the .zip
                        zip_ref.extractall(path=f'{file[:-4]}/')
                except zipfile.BadZipFile:
                    print(f"WARNING! Cannot extract {file}, not a valid ZIP archive.")
                # Remove the .zip file in favour of the extracted product
                os.remove(file)
                print(f"{file} has been unzipped and deleted.")
            else:
                print(f"{file} does not exist.")

        # Delete auxiliary files (trailers, quicklooks, manifests) not needed by most users.
        print('=' * 80)
        print('Deleting auxiliary files and subdirectories..')
        print('-' * 80)
        delete_auxiliary_files(out_dir)

    return



def Main():
    # Firstly, args common to all functions. 
    common = argparse.ArgumentParser(description='''Caller for downloading LI-L2-LEF, LI-L2-LGR or 
        LI-L2_LFL data files from EUMETSAT Data Store.''')
    common.add_argument('dataset', help='''Dataset to download: l2lfl for L2-LFL (flashes), l2lgr for L2-LGR 
        (groups), l2lef for L2-LEF (lightning events in flashes).''', type=str, 
        choices=['l2lfl', 'l2lgr', 'l2lef'])
    common.add_argument('period', help='''Period to download in UTC, can be a day like '20250304', an hour 
        like '2025030410', a minute like 202503041033, a second like 20250304103308 or start (included) and end
        (excluded) time of a period like 202503041255..202403041305''', type=str)
    common.add_argument('--out_dir', help='''Target directory for saving the downloaded LI data files, 
        default ./downloaded_data.''', type=str, default='./downloaded_data')
    args = common.parse_args()

    start, end = period_start_end(args.period)

    # Call the downloader.
    print(f'LI data downloader started at {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.')
    print(f"Downloading {cfg['products'][args.dataset]['fn_pattern']} files during {args.period}.")
    download_eum_data(args.dataset, start, end, args.out_dir)


if __name__ == '__main__':
    Main()
