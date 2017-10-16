
from __future__ import print_function
import zipfile
import os
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
def download_dataset():
    dataset_folder = os.path.dirname(os.path.abspath(__file__))
    if not (os.path.exists(os.path.join(dataset_folder, "positive")) and os.path.exists(os.path.join(dataset_folder, "positive")) and os.path.exists(os.path.join(dataset_folder, "positive"))):
        filename = os.path.join(dataset_folder, "HotailorPOC2.zip")
        if not os.path.exists(filename):
            url = "https://privdatastorage.blob.core.windows.net/github/cntk-hotel-pictures-classificator/HotailorPOC2.zip"
            print('Downloading data from ' + url + '...')
            urlretrieve(url, filename)
            
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(dataset_folder)
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + dataset_folder)
    
if __name__ == "__main__":
    download_dataset()