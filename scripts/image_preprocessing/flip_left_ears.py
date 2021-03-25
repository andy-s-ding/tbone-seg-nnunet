import ants
import numpy as np
import sys
import os
import glob 
import tqdm

def main(argv):
    # Read in data dir
    input_path = argv[0]
    output_path = argv[1]
    
    # Get all nifti file names
    in_path_list = glob.glob(input_path + "/*.nii.gz")
    
    for path in tqdm.tqdm(in_path_list):
        # If left ear, flip to right ear:
        filename = os.path.basename(path)
        if 'l' in os.path.basename(path.lower()):
            img = ants.image_read(path)
            img = img[::-1,:,:]
            img = ants.from_numpy(img)
            new_filename = filename.split('.nii.gz')[0] + "_flipped.nii.gz"
            print(f"Saving out {filename} as {new_filename}")
            ants.image_write(img, os.path.join(output_path, new_filename))
        

if __name__ == '__main__':
    main(sys.argv[1:])
    # example usage:  python flip_left_ears.py /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/ /scratch/users/jsoong1@jhu.edu/cis_ii/flipped_left_ears/

