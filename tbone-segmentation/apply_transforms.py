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
    parser.add_argument('--template', 
                        action="store", 
                        type=str, 
                        default="/media/andyding/SAMSUNG 4TB/tbone-seg-nnunet/registered_niftis/reg_RT_153.nii.gz"
                        )
    parser.add_argument('--target',
                        action="store",
                        type=str
                       )
    parser.add_argument('--transforms',
                        type=str,)
                        # nargs='+')
    parser.add_argument('--flip', 
                        action="store_true"
                        )
    parser.add_argument('--segmentations', 
                        action="store_true"
                        )
    parser.add_argument('--save_dir', 
                        action="store", 
                        type=str
                        )
    args = vars(parser.parse_args())
    return args

def main():
        gc.collect()
        args = parse_command_line(sys.argv)
        print(args)

        target_image = ants.image_read(args['target'])
        template_image = ants.image_read(args['template'])
        template_header = nrrd.read_header(args['template'])
        transform_list = args['transforms']

        if args['save_dir']: save_dir = args['save_dir']
        else: save_dir = os.path.dirname(args['target'])

        if args['flip']:
            if args['segmentations']: target_image = flip_image(target_image, single_components=True)
            else: target_image = flip_image(target_image)

        if args['segmentations']:
            individual_segments = ants.split_channels(target_image)
            print(len(individual_segments))
            transformed_targets = []
            for i, segment in enumerate(individual_segments): 

                print("applying transformation to channel %d"%i)

                print(segment.components)
                print(segment.pixeltype)
                print(segment.shape)

                transformed_target = ants.apply_transforms(fixed=template_image, moving=segment, transformlist=transform_list, interpolator='genericLabel', verbose=True)
                transformed_targets.append(transformed_target)

            transformed_target_image = ants.merge_channels(transformed_targets)

        else: 
            transformed_target_image = ants.apply_transforms(fixed=template_image, 
                                                             moving=target_image,                                 
                                                             transformlist=transform_list,
                                                             interpolator='linear',
                                                             verbose=True)


        print(transformed_target_image.shape)
        print("writing out registered images")

        # Save registered image
        ants_image_to_file(transformed_target_image, template_header, template_header, os.path.join(save_dir, "reg_" + os.path.basename(args['target'])), segmentations=args['segmentations'], nifti=False)
        # ants.image_write(transformed_target_image, os.path.join(save_dir, "reg_" + os.path.basename(args['target'])))

if __name__ == '__main__':
    main()

# example usage:
# python apply_transforms.py --template "/media/andyding/EXTREME SSD/ANTs-registration/segmentations/Segmentation RT 153.seg.nrrd" --target "/media/andyding/EXTREME SSD/ANTs-registration/segmentations/Segmentation RT 152.seg.nrrd" --transforms "/media/andyding/SAMSUNG 4TB/tbone-seg-nnunet/registered_niftis/transforms/RT_152_affine/tmphrx3urc00GenericAffine.mat" --segmentations