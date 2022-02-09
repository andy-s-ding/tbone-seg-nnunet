# jsad-tbone

Instructions for Temporal Bone Dataset Use:

## Step 0: Clone github through
```
git clone https://github.com/andy-s-ding/jsad-tbone
```

## Step 1: Set up two environments using .yml files in environments/
Navigate to environments/ folder

### Step 1.1: Creating scripting environment
Create environment from scripting_environment.yml file:
```
conda env create -f scripting_environment.yml
```


### Step 1.2: Creating nnUNet environment.
Create environment from nnUNet_environment.yml file:
```
conda env create -f nnUNet_environment.yml
```

## Step 2: Register data to some template.
Activate scripting environment
```
cd scripts/image_preprocessing
conda activate cis-ii
```
Register data to template
```
python generate_registration_sh.py <path to nifti images dir> <path to nifti segmentations dir> <path to output dir>
```
For internal use, this works:
```
python generate_registration_sh.py ../../nii_files/20210404 ../../NIFTI_Segmentations/20210404 <OUTPUT DIR>
```

## Step 3: Create datasplit for training/testing. Validation will automatically be chosen. 

For the general dataset, this is done by:
```
cd <path to github>/jsad-tbone/scripts/
python create_datasplit.py
```
For the SSM generated dataset, this is done by:
```
python create_generated_datasplit.py
```
Note that in order to create the generated datasplit, the general datasplit.pkl file needs to exist first. This is because the generated datasplit uses the same test set as the general split.

## Step 4: Create file structure required for nnUNet github. 
For the general temporal bone dataset:
```
python create_nnunet_filestructure.py --dataset original --original_dataset_dir <registered original images> --output_dir <output_dir> --pickle_path ./datasplit.pkl, --task_num <task num>```
```
For the SSM generated datasplit, this is done by:
```
python create_nnunet_filestructure.py --dataset generated --original_dataset_dir <registered original images dir> --generated_dataset_dir <generated dataset dir>--generated_label_dir <dir to SSM labels> --output_dir <output dir> --pickle_path ./datasplit_generated.pkl --task_num <task num>
```

## Step 5: Setup bashrc.
```
export nnUNet_raw_data_base="<PATH TO FILESTRUCTURE>/nnUnet/nnUNet_raw_data_base" 
export nnUNet_preprocessed="<PATH TO FILESTRUCTURE>/nnUNet_preprocessed" 
export RESULTS_FOLDER="<PATH TO FILESTRUCTURE>nnUnet/nnUNet_trained_models"
```
After updating this you will need to source your bashrc file.
```
source ~/.bashrc
```
This will deactivate your conda environment.

## Step 6: Verify and preprocess data.
Activate scripting environment
```
conda activate nnUNet-11.2
```

Run nnUNet preprocessing script
```
nnUNet_plan_and_preprocess -t <task_num> --verify_dataset_integrity
```
Note: You will freeze up if you don't have enough CPU! For MARCC usage, -n 12 in an interactive node is sufficient to complete in a reasonable amount of time
```
interact -p shared -c 12 -t 3:0:0
```
Potential Error: You may need to edit the dataset.json file so that the labels are sequential. Doing this in a text editor is completely fine!

## Step 7: Begin Training.
```
nnUNet_train 3d_fullres nnUNetTrainerV2 TaskXXX_TemporalBone Y --npz 
```
XXX refers to <task_num>

Y refers to the number of folds for cross-validation. If Y is set to "all" then all of the data will be used for training.
