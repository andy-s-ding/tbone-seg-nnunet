"""
deform_volumes.py

Andy's copy of registration_pipeline with functionality for volume deformation

"""
import os 
import sys 
import argparse 
import numpy as np
import ants
from ants.utils import convert_nibabel as cn
import nibabel as nib
import nrrd

import shutil
import psutil
import gc
import time

from utils.file_io import *
from utils.mask import flip_image
from propagate_segments import adjust_file_path

side = 'RT'
base = "/Volumes/Extreme SSD/ANTs-registration/"
template = '153'
target = '138'
save_dir = os.path.join(base, 'predictions')
downsample = True
downsample_size = 60
print('-- Reading NRRD')
# deformed_volume_path = adjust_file_path(save_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample, downsample_size=downsample_size, flip=flip)
# deformed_volume_path = '/Volumes/Extreme SSD/ANTs-registration/deformed_volumes/RT 153 deform1-downsample60.nrrd'
deformed_seg_path = '/Volumes/Extreme SSD/ANTs-registration/predictions/Segmentation RT 153 deform1-downsample60.seg.nrrd'
# ants_header = ants.image_header_info(deformed_seg_path)
ants_img = ants.image_read(deformed_seg_path)
data = ants_img.view(single_components=True)
# ants_header = ants.image_header_info(deformed_volume_path)
header = nrrd.read_header(deformed_seg_path)
data = convert_to_one_hot(data, header)
fg = np.max(data, axis=0)
labelmap = np.multiply(np.argmax(data, axis=0) + 1, fg).astype('uint8')
print(data.shape)
deformed_seg = ants.from_numpy(labelmap, origin=ants_img.origin, spacing = ants_img.spacing, direction=ants_img.direction)
# deformed_nii_path = adjust_file_path(save_dir, "Segmentation %s %s %s"%(side, template, target), "_withants.nii.gz", downsample=downsample, downsample_size=downsample_size)
deformed_nii_path = '/Volumes/Extreme SSD/ANTs-registration/predictions/NIFTI Predictions/Segmentation RT 153 deform1-downsample60_withANTs.nii.gz'
print('-- Saving NII')
deformed_seg.to_filename(deformed_nii_path)
