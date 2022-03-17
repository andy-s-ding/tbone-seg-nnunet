# python register.py ../nii_files/RT_153.nii.gz ../nii_files/LT_143.nii.gz ../NIFTI\ Segmentations/Segmentation_LT_143.nii.gz ../test_imgs

import glob
import os
import numpy as np
import sys

        
def main(argv):
    # Read in data dir
    image_path = argv[0]
    segmentation_path = argv[1]
    output_dir = argv[2]
    
    # Get all nifti file names
    in_seg_path_list = glob.glob(segmentation_path + "/*.nii.gz")
    print(f"There are {len(in_seg_path_list)} files to register.")

    # Prepare list to write to
    command_list = []
    string_base = "python register.py"
    
    # Template path
    template = 'RT_153.nii.gz'
    template_path = os.path.join(image_path, template)

    for seg_path in in_seg_path_list:
        try:
            seg_file_name = os.path.basename(seg_path)
            nii_file_name = seg_file_name.split('Segmentation_')[-1]
            command = string_base + " " +  template_path + " " + os.path.join(image_path, nii_file_name) + " " + seg_path + " " + output_dir
            command_list.append(command)
        except Exception as e:
            print("~"*10)
            print(e)
            print("~"*10)
        
    with open('run_register.sh', 'w') as f:
        for command in command_list:
            print(command)
            f.write("%s\n" % command)
        
if __name__ == '__main__':
    main(sys.argv[1:])
    # example usage:  python generate_registration_sh.py ../../nii_images ../../nii_segmentations/20220316_updated_gt_segmentations ../../registered_niftis
