python create_nnunet_filestructure.py --dataset generated \
--original_dataset_dir ../registered_niftis/ \
--generated_dataset_dir ../nii_files/20210411_ssm_data_downsample100/ \
--generated_label_dir ../NIFTI_Segmentations/20210419_updated_ssm_segmentations_downsample100 \
--output_dir ../02_jsoong/ \
--pickle_path ./Ablation_1/datasplit_generated.pkl \
--task_num 102
