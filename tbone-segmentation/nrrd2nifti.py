import os 
import sys 
import numpy as np
import ants
from ants.utils import convert_nibabel as cn
import nibabel as nib
import nrrd

import glob

side = 'RT'
base = "/Volumes/Extreme SSD/ANTs-registration/"
seg_dir = os.path.join(base, 'segmentations')
save_dir = os.path.join(seg_dir, 'NIFTI Segmentations')

segmentations = glob.glob(os.path.join(seg_dir, 'Segmentation *.seg.nrrd'))

for seg_path in segmentations:  
    print(seg_path)
    print('-- Reading NRRD')
    file_prefix = seg_path.split(os.path.sep)[-1].split('.seg.nrrd')[0]
    ants_img = ants.image_read(seg_path)
    data = ants_img.view(single_components=True)
    header = nrrd.read_header(seg_path)
    data = convert_to_one_hot(data, header)
    fg = np.max(data, axis=0)
    labelmap = np.multiply(np.argmax(data, axis=0) + 1, fg).astype('uint8')
    print(data.shape)
    seg_img = ants.from_numpy(labelmap, origin=ants_img.origin, spacing = ants_img.spacing, direction=ants_img.direction)
    deformed_nii_path = os.path.join(save_dir, file_prefix + '.nii.gz')
    print('-- Saving NII')
    print(deformed_nii_path)
    seg_img.to_filename(deformed_nii_path)
