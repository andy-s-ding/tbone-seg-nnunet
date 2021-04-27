import pickle
import random
import pandas as pd
import numpy as np
import glob
import os

random.seed(0)
np.random.seed(0)

SSM_data = glob.glob(os.path.join("../nii_files/20210411_ssm_data_downsample100", "*.nii.gz"))

SSM_data_ablation_1 = [SSM_data[i] for i in [random.randint(0, len(SSM_data)-1) for _ in range(12)]]

for file in SSM_data_ablation_1:
    # Copy
    command = f"cp {file} ../ablation_1/nii_files"
#     print(command)
    os.system(command)
    command = f"cp ../NIFTI_Segmentations/20210419_updated_ssm_segmentations_downsample100/Segmentation_{os.path.basename(file)} ../ablation_1/NIFTI_Segmentations"
#     print(command)
    os.system(command)