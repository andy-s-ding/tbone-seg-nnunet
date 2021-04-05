# jsad-tbone

Instructions for Temporal Bone Dataset Use:

Step 0: Clone github through
git clone https://github.com/jpsoong/jsad-tbone

Step 1: Set up environment through instructions on nnUNet, and activate anaconda environment
```
conda activate nnUNet
```
Step 2: Create datasplit for training/testing. Validation will automatically be chosen. 

For the general dataset, this is done by:
```
cd Scripts/
python create_datasplit.py
```
For the SSM generated dataset, this is done by:
```
python create_generated_datasplit.py
```
Note that in order to create the generated datasplit, the general datasplit.pkl file needs to exist first. This is because the generated datasplit uses the same test set as the general split.

Step 3: Create file structure required for nnUNet github. 
For the general temporal bone dataset:
```
python create_nnunet_filestructure.py --dataset original --original_dataset_dir <registered original images> --output_dir <output_dir> --pickle_path ./datasplit.pkl, --task_num <task num>```
```
For the SSM generated datasplit, this is done by:
```
python create_nnunet_filestructure.py --dataset generated --original_dataset_dir <registered original images dir> --generated_dataset_dir <generated dataset dir>--generated_label_dir <dir to SSM labels> --output_dir <output dir> --pickle_path ./datasplit_generated.pkl --task_num <task num>
```

Step 4: Train
