import os 
import sys 
import argparse 
import numpy as np
import pandas as pd
import nrrd

from utils.file_io import *

import glob

side = 'RT'
base = "/Volumes/Extreme SSD/ANTs-registration/"
seg_dir = os.path.join(base, 'segmentations')

segmentations = glob.glob(os.path.join(seg_dir, 'Segmentation *.seg.nrrd'))

seg_name_path = os.path.join(base, 'segment_names.csv')
seg_name_dict = dict()
for seg_path in segmentations:  
    print(seg_path)
    print('-- Reading NRRD Header')
    file_prefix = seg_path.split(os.path.sep)[-1].split('.seg.nrrd')[0]
    seg_names = get_segmentation_names(nrrd.read_header(seg_path))
    seg_name_dict[file_prefix] = seg_names

seg_name_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in seg_name_dict.items()]))
seg_name_df.to_csv(seg_name_path)
