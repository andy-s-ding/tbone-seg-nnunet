# python register.py ../nii_files/RT_153.nii.gz ../nii_files/LT_143.nii.gz ../NIFTI\ Segmentations/Segmentation_LT_143.nii.gz ../test_imgs

import glob
import os
import numpy as np
import sys

        
def main(argv):
    # Read in data dir
    input_path = argv[0]
    segmentation_path = argv[1]
    output_dir = argv[2]
    
    # Get all nifti file names
    in_seg_path_list = glob.glob(segmentation_path + "/*.nii.gz")
    in_seg_path_list = [path for path in in_seg_path_list if "153" not in path]
    print(f"There are {len(in_seg_path_list)} files to register.")

    # Prepare list to write to
    command_list = []
    string_base = "python register.py"
    
    # Template path
    template = '../../nii_files/20210404_images/RT_153.nii.gz'

    for path in in_seg_path_list:
        try:
            seg_file_name = os.path.basename(path)
            nii_file_name = seg_file_name.split('Segmentation_')[-1]
            
            command = string_base + " " +  template + " " + os.path.join(input_path, nii_file_name) + " " + path + " " + output_dir
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
    # example usage:  python generate_registration_sh.py ../../nii_files/20210404 ../../NIFTI_Segmentations/20210404 <OUTPUT DIR>
