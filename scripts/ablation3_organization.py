import os
import sys
import random
import pandas
import glob
from tqdm import tqdm

def main(argv):
    random.seed(0)
    vol_dir = argv[0]
    seg_dirs = argv[1]
    
    folder_names = [f'{i}_template' for i in range(1, 6)]
    template_files = []
    init_splits = {}
    
    for i in range(5):
        template_files.append(glob.glob(os.path.join(vol_dir, folder_names[i], "*.nii.gz")))
        init_splits[i] = template_files[i].copy()
        print(len(init_splits[i]))
    
    final_splits = {}
    # Skip template_1 since it is already done
    
    # template_2: Sample 60 files from template_files[0]
    tmp1 = random.sample(init_splits[0], 60)
    assert(len(tmp1) == 60)
    final_splits[folder_names[1]] = tmp1 # + init_splits[1]
    
    # template 3: Sample 40 files from template_files[1], template_files[0]
    tmp1 = random.sample(init_splits[0], 40)
    tmp2 = random.sample(init_splits[1], 40)
    assert(len(tmp1) == 40)
    assert(len(tmp2) == 40)
    final_splits[folder_names[2]] = tmp1 + tmp2 # init_splits[2]
    
    # template 3: Sample 30 files from template_files[1], template_files[0]
    tmp1 = random.sample(init_splits[0], 30) 
    tmp2 = random.sample(init_splits[1], 30)
    tmp3 = random.sample(init_splits[2], 30)
    assert(len(tmp1) == 30)
    assert(len(tmp2) == 30)
    assert(len(tmp3) == 30)
    final_splits[folder_names[3]] = tmp1 + tmp2 + tmp3 # + init_splits[3]
    
    # template 4: Sample 24 files from template_files[1], template_files[0]
    tmp1 = random.sample(init_splits[0], 24) 
    tmp2 = random.sample(init_splits[1], 24) 
    tmp3 = random.sample(init_splits[2], 24)
    tmp4 = random.sample(init_splits[3], 24)
    assert(len(tmp1) == 24)
    assert(len(tmp2) == 24)
    assert(len(tmp3) == 24)
    assert(len(tmp4) == 24)
    final_splits[folder_names[4]] = tmp1 + tmp2 + tmp3 + tmp4 # + init_splits[4] 
    
    for split in final_splits:
        print(len(final_splits[split]))
    
    print("="*10)
    print("Starting Moving")
    print("="*10)
    
    # Now move files...
    for split in final_splits: 
        for vol in tqdm(final_splits[split]):
            # do image
            src = vol
            dst = os.path.join(vol_dir, split, os.path.basename(vol))
            command = f"cp {src} {dst}"
#             print(command)
            os.system(command)
            
#             breakpoint()
            # do label
            seg_split = os.path.basename(os.path.dirname(vol))
            if '153' in vol:
                src = os.path.join(seg_dirs, seg_split, "Segmentation_" + os.path.basename(vol))
                dst = os.path.join(seg_dirs, split, "Segmentation_" + os.path.basename(vol))
            else:    
                src = os.path.join(seg_dirs, seg_split, "reg_Segmentation_" + os.path.basename(vol).split('reg_')[-1])
                dst = os.path.join(seg_dirs, split, "reg_Segmentation_" + os.path.basename(vol).split('reg_')[-1])
            command = f"cp {src} {dst}"
#             print(command)
            os.system(command)
#             breakpoint()
            
            
if __name__ == '__main__':
    main(sys.argv[1:])    
    
    # usage: python ablation3_organization.py ../nii_files/Ablation_3 ../NIFTI_Segmentations/Ablation_3
