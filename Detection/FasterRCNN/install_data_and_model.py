# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import zipfile
import os, sys
from config import cfg



def create_mappings(folder_path):
    sys.path.append(os.path.join(folder_path, "..", "..",  "Detection", "utils", "annotations"))
    from annotations_helper import create_class_dict, create_map_files
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(abs_path, cfg["CNTK"].MAP_FILE_PATH)

    class_dict = create_class_dict(data_set_path)
    create_map_files(data_set_path, class_dict, training_set=True)
    create_map_files(data_set_path, class_dict, training_set=False)

if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.abspath(__file__))

    #downloads pretrained model pointed out in config.py that will be used for transfer learning
    sys.path.append(os.path.join(base_folder, "..", "..",  "PretrainedModels"))
    from models_util import download_model_by_name
    download_model_by_name(cfg["CNTK"].BASE_MODEL)

    #downloads hotel pictures classificator dataset (HotailorPOC2)
    sys.path.append(os.path.join(base_folder, "..", "..",  "DataSets", "HotailorPOC2"))
    from download_HotailorPOC2_dataset import download_dataset
    download_dataset()

    #generates metadata for dataset required by FasterRCNN.py script
    print("Creating mapping files for data set..")
    create_mappings(base_folder)

