# tbone-seg-nnunet

Instructions for Temporal Bone Dataset Use:

## Step 0: Fork this GitHub repo
```
git clone https://github.com/<your-username>/tbone-seg-nnunet
```

## Step 1: Set up two environments using .yml files in environments/
Navigate to environments/ folder

### Step 1.1: Creating scripting and nnUNet environments
Create environment from scripting_environment.yml file:
```
conda env create -f scripting_environment.yml
```
Create environment from nnUNet_environment.yml file:
```
conda env create -f nnUNet_environment.yml
```
For internal use, the scripting environment is named `cis-ii` and the nnUNet environment is named `nnUNet-11.2`.

Environment names can be changed in the `.yml` files.

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

For the general (default) dataset without deformation field SSM augmentation, this is done by:
```
cd <path to github>/tbone-seg-nnunet/scripts/
python create_datasplit.py
```
For the deformation field SSM-augmented (generated) dataset, this is done by:
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
Edit your `~/.bashrc` file with `gedit ~/.bashrc` or `nano ~/.bashrc`. At the end of the file, add the following lines:
```
export nnUNet_raw_data_base="<PATH TO FILESTRUCTURE>/nnUnet/nnUNet_raw_data_base" 
export nnUNet_preprocessed="<PATH TO FILESTRUCTURE>/nnUNet_preprocessed" 
export RESULTS_FOLDER="<PATH TO FILESTRUCTURE>nnUnet/nnUNet_trained_models"
```
After updating this you will need to source your `~/.bashrc` file.
```
source ~/.bashrc
```
This will deactivate your current conda environment.

## Step 6: Verify and preprocess data.
Activate scripting environment.
```
conda activate nnUNet-11.2
```
Run nnUNet preprocessing script.
```
nnUNet_plan_and_preprocess -t <task_num> --verify_dataset_integrity
```
Note: You will freeze up if you don't have enough CPU! For MARCC usage, `-n 12` in an interactive node is sufficient to complete in a reasonable amount of time:
```
interact -p shared -c 12 -t 3:0:0
```
Potential Error: You may need to edit the dataset.json file so that the labels are sequential. If you have at least 10 labels, then labels `10, 11, 12,...` will be arranged before labels `2, 3, 4, ...`. Doing this in a text editor is completely fine!

## Step 7: Begin Training.
```
nnUNet_train 3d_fullres nnUNetTrainerV2 Task<task_num>_TemporalBone Y --npz
```
`Y` refers to the number of folds for cross-validation. If `Y` is set to `all` then all of the data will be used for training.

`--npz` makes the models save the softmax outputs (uncompressed, large files) during the final validation. It should only be used if you are training multiple configurations, which requires `nnUNet_find_best_configuration` to find the best model. We omit this by default.
