rm -r ../test_jsoong
mkdir ../test_jsoong
python create_nnunet_filestructure.py \
 --dataset generated \
 --original_dataset_dir ../registered_niftis/ \
 --generated_dataset_dir ../nii_files \
 --generated_label_dir ../NIFTI_Segmentations \
 --output_dir ../test_jsoong/ \
 --pickle_path ./datasplit_generated.pkl \
 --task_num 999