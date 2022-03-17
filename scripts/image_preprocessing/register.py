"""
register.py
Application of ANTS to register target and target labels to template 
"""
import os 
import sys 
import argparse 
import numpy as np
import ants

import shutil
import psutil
import gc

# from utils.mask import flip_image

def flip_image(ants_image, axis=0, single_components=False): 
    data = ants_image.numpy(single_components=single_components)
    flipped_data = np.flip(data, axis=axis)
    if flipped_data.dtype == 'int16':
        flipped_data = flipped_data.astype('float32')
    flipped_data = flipped_data.squeeze()
    return ants_image.new_image_like(flipped_data)

def register_to_target(template_path, target_path, target_segmentation_path, side, save_dir, dry=False, downsample=False, downsample_size=300, write_transforms=False): 
    """Summary

    Args:
        template (TYPE): Description
        target (TYPE): Description
        base (TYPE): Description
        side (TYPE): Description
        save_dir (TYPE): Description
        dry (bool, optional): Description
        downsample (bool, optional): Description
        downsample_size (int, optional): Description
        write_transforms (bool, optional): Description
        write_annotations (bool, optional): Description

    Returns:
        TYPE: Description
    """
    print("---"*10)
    print("entering registration w/ template %s and target %s"%(template_path, target_path))

    if side == 'RT': 
        other_side = 'LT'
        flip = False 
    else: 
        other_side = 'RT'
        flip = True
    
    if dry: 
        pass

    if not dry: 
        
        target_image = ants.image_read(target_path)
        template_image = ants.image_read(template_path)
        target_segmentations = ants.image_read(target_segmentation_path, pixeltype="unsigned char")

        if flip:
            target_image = flip_image(target_image)
            target_segmentations = flip_image(target_segmentations, single_components=True)

        if downsample:
            target_image_downsample = ants.resample_image(target_image, (downsample_size, downsample_size, downsample_size), 1, 0)
            template_image_downsample = ants.resample_image(template_image, (downsample_size, downsample_size, downsample_size), 1, 0)
            transform_forward = ants.registration(fixed=template_image_downsample, moving=target_image_downsample, type_of_transform="Affine", verbose=True)

        else:
            transform_forward = ants.registration(fixed=template_image, moving=target_image, type_of_transform="Affine", verbose=True)


        registered_target_image = ants.apply_transforms(fixed=template_image, 
                                                        moving=target_image,                                 
                                                        transformlist=transform_forward['fwdtransforms'],
                                                        interpolator='linear')
        registered_target_segmentations = ants.apply_transforms(fixed=template_image,
                                                                moving=target_segmentations,
                                                                transformlist=transform_forward['fwdtransforms'],
                                                                interpolator='genericLabel')
 
        print(registered_target_segmentations.shape)
        print("writing out registered images")

        # Save registered image
        ants.image_write(registered_target_image, os.path.join(save_dir, "reg_" + os.path.basename(target_path)))
    
        # Save registered labels
        ants.image_write(registered_target_segmentations, os.path.join(save_dir, "reg_" + os.path.basename(target_segmentation_path)))

        if write_transforms:
            print("writing transforms")
            transform_dir = os.path.join(save_dir, "transforms")
            folder_name = os.path.basename(target_path).split('.nii.gz')[0]

            affine_path = os.path.join(transform_dir, folder_name + "_affine")
            
            try:
                os.mkdir(affine_path)
            except:
                print(f"{affine_path} already exists") 
            
            shutil.move(transform_forward['fwdtransforms'][0], affine_path)

        gc.collect()

    return 

def main(argv):
    template_path = argv[0]
    target_path = argv[1]
    target_segmentation_path = argv[2]
    save_dir = argv[3]
    if "LT" in target_path.upper(): side='LT'
    else: side='RT'
    
    register_to_target(template_path, target_path, target_segmentation_path, side, save_dir, dry=False, downsample=False, downsample_size=300, write_transforms=True)


if __name__ == '__main__':
    main(sys.argv[1:])
# example usage:  python register.py ../nii_files/RT_153.nii.gz ../right_ears/RT_142.nii.gz ../NIFTI\ Segmentations/Segmentation_RT_142.nii.gz ../test_imgs

# python register.py ../nii_files/RT_153.nii.gz ../nii_files/LT_143.nii.gz ../NIFTI\ Segmentations/Segmentation_LT_143.nii.gz ../test_imgs
