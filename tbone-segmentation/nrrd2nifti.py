import os 
import sys 
import numpy as np
import ants
from ants.utils import convert_nibabel as cn
import nibabel as nib
import nrrd

from utils.file_io import *

import glob

side = 'RT'
# base = "/Volumes/Extreme SSD/ANTs-registration/"
base = "/media/andyding/172A-29C2/ANTs-registration/"
seg_dir = os.path.join(base, 'segmentations')
save_dir = os.path.join(seg_dir, 'NIFTI Segmentations')

segmentations = glob.glob(os.path.join(seg_dir, 'Segmentation *.seg.nrrd'))

for seg_path in segmentations:  
    print(seg_path)
    print('-- Reading NRRD')
    file_prefix = seg_path.split(os.path.sep)[-1].split('.seg.nrrd')[0]
    ants_img = ants.image_read(seg_path)
    header = nrrd.read_header(seg_path)
    deformed_nii_path = os.path.join(save_dir, file_prefix + '.nii.gz')
    print('-- Saving NII')
    print(deformed_nii_path)
    ants_image_to_file(ants_img, header, header, deformed_nii_path, nifti=True)
