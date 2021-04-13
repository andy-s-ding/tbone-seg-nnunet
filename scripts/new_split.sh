python create_nnunet_filestructure.py --dataset generated \
--original_dataset_dir ../registered_niftis/ \
--generated_dataset_dir ../nii_files/20210411_ssm_data_downsample100/ \
--generated_label_dir ../NIFTI_Segmentations/20210411_ssm_segmentations_downsample100/ \
--output_dir ../01_ading/ \
--pickle_path ./datasplit_generated.pkl \
--task_num 101