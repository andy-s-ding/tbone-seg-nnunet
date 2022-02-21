"""
apply_transforms.py
Applies transforms to an image
"""
from ast import AsyncFunctionDef
import os 
import sys 
import argparse 
import numpy as np
import ants
import nrrd

import shutil
import psutil
import gc

from utils.mask import flip_image
from utils.transforms import apply_transform_to_image
from utils.file_io import *


def parse_command_line(args):
    print('parsing command line')

    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
    parser.add_argument('--fixed', 
                        action="store", 
                        type=str, 
                        default="/media/andyding/SAMSUNG 4TB/tbone-seg-nnunet/registered_niftis/reg_RT_153.nii.gz",
                        help="Reference image (NRRD format) used for spacial information after applying transforms on the moving image."
                        )
    parser.add_argument('--moving',
                        action="store",
                        type=str,
                        help="Input image (NRRD or NIFTI format) to transform."
                       )
    parser.add_argument('--transforms',
                        type=str,
                        nargs='+',
                        help="List of paths to transforms that will be applied to the moving image."
                        )
    parser.add_argument('--flip', 
                        action="store_true",
                        help="True if fixed and moving images are not on the same side."
                        )
    parser.add_argument('--segmentations', 
                        action="store_true",
                        help="True if moving image is a segmentation image and not a volume image."
                        )
    parser.add_argument('--save_dir', 
                        action="store", 
                        type=str,
                        help="Output directory to save transformed image. If none specified, the parent directory of the moving image will be used."
                        )
    parser.add_argument('--prefix', 
                        action="store", 
                        type=str,
                        help="Prefix appended to the moving image filename. If none specified, the prefix will be 'reg_'."
                        )
    args = vars(parser.parse_args())
    return args

def main():
        args = parse_command_line(sys.argv)
        print(args)

        moving_image = ants.image_read(args['moving'])
        fixed_image = ants.image_read(args['fixed'])

        # We only need the spacial information of the fixed image. Multiple components result in seg fault
        if fixed_image.components > 1:
            fixed_image = ants.split_channels(fixed_image)[0]
        
        moving_header = nrrd.read_header(args['moving'])
        fixed_header = nrrd.read_header(args['fixed'])
        transform_list = args['transforms']

        if args['save_dir']: save_dir = args['save_dir']
        else: save_dir = os.path.dirname(args['moving'])

        if args['flip']:
            if args['segmentations']: moving_image = flip_image(moving_image, single_components=True)
            else: moving_image = flip_image(moving_image)

        if args['segmentations']:
            individual_segments = ants.split_channels(moving_image)
            print(len(individual_segments))
            transformed_segments = []
            for i in range(len(individual_segments)): 

                print("applying transformation to channel %d"%i)
                segment = individual_segments[i]
                transformed_moving = ants.apply_transforms(fixed=fixed_image, moving=segment, transformlist=transform_list, interpolator='genericLabel', verbose=True)
                transformed_segments.append(transformed_moving)

            transformed_moving_image = ants.merge_channels(transformed_segments)

        else: 
            transformed_moving_image = ants.apply_transforms(fixed=fixed_image, 
                                                             moving=moving_image,                                 
                                                             transformlist=transform_list,
                                                             interpolator='linear',
                                                             verbose=True)


        print(transformed_moving_image.shape)
        print("writing out registered images")

        # Save registered image
        if args['prefix']: prefix = args['prefix']
        else: prefix = "reg_"
        ants_image_to_file(transformed_moving_image, moving_header, fixed_header, os.path.join(save_dir, prefix + os.path.basename(args['moving'])), segmentations=args['segmentations'], nifti=False)

if __name__ == '__main__':
    main()

# example usage:
# python apply_transforms.py --fixed "/media/andyding/EXTREME SSD/ANTs-registration/images/RT 153.nrrd" --moving "/media/andyding/EXTREME SSD/ANTs-registration/segmentations/Segmentation RT 152.seg.nrrd" --transforms "/media/andyding/SAMSUNG 4TB/tbone-seg-nnunet/registered_niftis/transforms/RT_152_affine/tmphrx3urc00GenericAffine.mat" --segmentations