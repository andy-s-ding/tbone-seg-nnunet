# env FSLOUTPUTTYPE=NIFTI_GZ flirt -dof 7 -cost corratio -searchcost corratio -searchrx –10 10 -searchry –10 10 -searchrz –10 10 -in /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/LT 143.nii.gz -ref /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/RT 153.nii.gz -omat /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/LT 143.omat -out /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/LT 143_flirt.nii.gz -v
import glob
import os
import numpy as np
import sys

        
def main(argv):
    # Read in data dir
    input_path = argv[0]
    output_path = argv[1]
    # Get all nifti file names
    in_path_list = glob.glob(input_path + "/*.nii.gz")
    out_path_list = glob.glob(output_path + "/*.nii.gz")

    # Prepare list to write to
    command_list = []
    flirt_string_base = "env FSLOUTPUTTYPE=NIFTI_GZ flirt -dof 7 -cost corratio -searchcost corratio -searchrx -10 10 -searchry -10 10 -searchrz -10 10 "
    
    # Template path
    template = '-ref /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/RT_153.nii.gz '

    for path in in_path_list:
#         print(path)
        input_file = os.path.basename(path)
        output_file = input_file.split(".nii.gz")[0] + "_flirt.nii.gz "
        output_omat = output_file.split('.nii.gz')[0] + ".omat "
        input_string = "-in " + input_path + input_file + " "
        output_string = "-out " + output_path + output_file + " "
        output_omat_string = '-omat ' + output_path + output_omat + " "
        command = flirt_string_base + input_string + template + output_omat_string + output_string + "-verbose 2"
#         print(command)
        command_list.append(command)
    
    with open('run_flirt.sh', 'w') as f:
        for command in command_list:
            print(command)
            f.write("%s\n" % command)
        
if __name__ == '__main__':
    main(sys.argv[1:])
    # example usage:  python generate_flirt_sh.py /scratch/users/jsoong1@jhu.edu/cis_ii/nii_files/ /scratch/users/jsoong1@jhu.edu/cis_ii/fsl_registered/
