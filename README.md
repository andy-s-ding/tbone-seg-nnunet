# tbone-seg-nnunet

**Important:** This pipeline has been validated for Ubuntu 18.04 LTS and CUDA 11.2. Other Linux distros or versions of Ubuntu, as well as versions of CUDA may not be compatible with this repo out of the box.

Instructions for Temporal Bone Dataset Use:

## Step 0: Fork this GitHub repo
```
git clone https://github.com/<your-username>/tbone-seg-nnunet
```

## Step 1: Set up scripting and nnUNet environments
Navigate to the `environments` folder:
```
cd <path to github>/tbone-seg-nnunet/environments
```
Create environment from scripting_environment.yml file:
```
conda env create -f scripting_environment.yml
```
Create environment from nnUNet-cuda-11.2_environment.yml file:
```
conda env create -f nnUNet-cuda-11.2_environment.yml
```
For internal use, the scripting environment is named `cis-ii` and the nnUNet environment is named `nnUNet-11.2`. The scripting environment is used for setting up the file structure for nnUNet training while the nnUNet environment is used for nnUNet training, validation, and inference.

Environment names can be changed in the `.yml` files.

## Step 2: Register data to some template
Image registration is done in the `image_preprocessing` folder:
```
cd <path to github>/tbone-seg-nnunet/scripts/image_preprocessing
```
Activate the scripting environment:
```
conda activate cis-ii
```
Images and their segmentations (labels) are in .`nii.gz` format. The naming convention for images is `<LT/RT>_<ID>.nii.gz` while the naming convention for segmentations is `Segmentation_<LT/RT>_<ID>`.

`tbone-seg-nnunet/scripts/register.py` is a subroutine that registers one target image and its segmentations to a designated template image. `generate_registration_sh.py` will create a `.sh` script that will iteratively call `register.py` to co-register all target images to a designated template (currently hard-coded). Within `generate_registration_sh.py`, first set `template` to be the basename of the image file you would like to use as the template. Then run the following code:
```
python generate_registration_sh.py <path to nifti images dir> <path to nifti segmentations dir> <path to output dir>
```
For internal use, this works:
```
python generate_registration_sh.py ../../nii_images ../../nii_segmentations ../../registered_niftis
```
We highly recommend checking the last registered image, since data may not be completely written. If this is the case, run `register.py` for just the last image.

## Step 3: Create datasplit for training/testing. Validation will automatically be chosen
Navigate to the `scripts` folder:
```
cd <path to github>/tbone-seg-nnunet/scripts/
```

The datasplit file will be a `.pkl` file that will be referenced when creating the final file structure for nnUNet training.

For the general (default) dataset without deformation field SSM generation, this is done by:
```
python create_datasplit.py
```
For the deformation field SSM-generated dataset, this is done by:
```
python create_generated_datasplit.py
```
Note that in order to create the SSM-generated datasplit, the general `datasplit.pkl` file needs to exist first. This is because the generated datasplit uses the same test set as the general split.

## Step 4: Create file structure required for nnUNet
Create a base directory `tbone-seg-nnunet/<BASE_DIR>` that will serve as the root directory for the nnUNet training file structure.

In the `scripts/` folder, run `create_nnunet_filestructure.py` to copy training and test data over based on the datasplit `.pkl` generated in Step 3.

For the general temporal bone dataset:
```
python create_nnunet_filestructure.py --dataset original --original_dataset_dir <registered original images> --output_dir <BASE_DIR> --pickle_path ./datasplit.pkl, --task_num <task num>
```
For the SSM generated datasplit, this is done by:
```
python create_nnunet_filestructure.py --dataset generated --original_dataset_dir <registered original images dir> --generated_dataset_dir <generated dataset dir>--generated_label_dir <dir to SSM labels> --output_dir <BASE_DIR> --pickle_path ./datasplit_generated.pkl --task_num <task num>
```

## Step 5: Setup bashrc
Edit your `~/.bashrc` file with `gedit ~/.bashrc` or `nano ~/.bashrc`. At the end of the file, add the following lines:
```
export nnUNet_raw_data_base="<ABSOLUTE PATH TO BASE_DIR>/nnUnet/nnUNet_raw_data_base" 
export nnUNet_preprocessed="<ABSOLUTE PATH TO BASE_DIR>/nnUNet_preprocessed" 
export RESULTS_FOLDER="<ABSOLUTE PATH TO BASE_DIR>/nnUnet/nnUNet_trained_models"
```
After updating this you will need to source your `~/.bashrc` file:
```
source ~/.bashrc
```

## Step 6: Verify and preprocess data
Activate the nnUNet environment:
```
conda activate nnUNet-11.2
```
Run the nnUNet preprocessing script:
```
nnUNet_plan_and_preprocess -t <task_num> --verify_dataset_integrity
```
Note: You will freeze up if you don't have enough CPU! For MARCC usage, `-n 12` in an interactive node is sufficient to complete in a reasonable amount of time:
```
interact -p shared -c 12 -t 3:0:0
```
Potential Error: You may need to edit the dataset.json file so that the labels are sequential. If you have at least 10 labels, then labels `10, 11, 12,...` will be arranged before labels `2, 3, 4, ...`. Doing this in a text editor is completely fine!

## Step 7: Begin Training
For vanilla training on a 3D nnUNet, run:
```
nnUNet_train 3d_fullres nnUNetTrainerV2 Task<task_num>_TemporalBone Y --npz
```
`Y` refers to the number of folds for cross-validation. If `Y` is set to `all` then all of the data will be used for training.

`--npz` makes the models save the softmax outputs (uncompressed, large files) during the final validation. It should only be used if you are training multiple configurations, which requires `nnUNet_find_best_configuration` to find the best model. We omit this by default.

Variants of the `nnUNetTrainerV2` class can be made and saved in `tbone-seg-nnunet/nnUNet/nnunet/training/network_training`. Refer to other variants in the `network_training/nnUNet_variants` folder for examples. Training with a custom `nnUNetTrainerV2` variant can then be run as:

```
nnUNet_train 3d_fullres <nnUNetTrainerV2 Variant Name> Task<task_num>_TemporalBone Y --npz
```
Multiple variants can be trained on the same dataset.
