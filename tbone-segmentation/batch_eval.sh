## batch evaluation
#!/bin/bsh

base="/Users/alex/Documents/segmentations/"
predictions="/Users/alex/Documents/predictions/"

# base="/scratch/groups/rtaylor2/ANTs-registration/segmentations"
# predictions="/scratch/groups/rtaylor2/ANTs-registration/predictions"

# # evaluate 153 -> 138
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 138.seg.nrrd --prediction "${predictions}RT 153 138-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 138" --prefix_pred "RT_153_138"

# # evaluate 153 -> 142 
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 142.seg.nrrd --prediction "${predictions}RT 153 142-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 142" --prefix_pred "RT_153_142"

# # evaluate 153 -> 143
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 143.seg.nrrd --prediction "${predictions}RT 153 143-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 143" --prefix_pred "RT_153_143"

# # evaluate 153 -> 144
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 144.seg.nrrd --prediction "${predictions}RT 153 144-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 144" --prefix_pred "RT_153_144"

# # # evaluate 153 -> 145 
# # python3 eval.py --ground_truth ${base}Segmentation\ RT\ 145.seg.nrrd --prediction "${predictions}RT 153 145-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 145" --prefix_pred "RT_153_145"

# # evaluate 153 -> 146 
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 146.seg.nrrd --prediction "${predictions}RT 153 146-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 146" --prefix_pred "RT_153_146"

# # evaluate 153 -> 147 
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 147.seg.nrrd --prediction "${predictions}RT 153 147-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 147" --prefix_pred "RT_153_147"

# # # evaluate 153 -> 150
# # python3 eval.py --ground_truth ${base}Segmentation\ RT\ 150.seg.nrrd --prediction "${predictions}RT 153 150-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 150" --prefix_pred "RT_153_150"

# # evaluate 153 -> 152
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 152.seg.nrrd --prediction "${predictions}RT 153 152-syn80-demons.seg.nrrd" --hausdorff --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 152" --prefix_pred "RT_153_152"

########## 

# evaluate 153 -> 138
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 138.seg.nrrd --prediction "${predictions}RT 153 138-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 138" --prefix_pred "RT_153_138"

# evaluate 153 -> 142 
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 142.seg.nrrd --prediction "${predictions}RT 153 142-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 142" --prefix_pred "RT_153_142"

# evaluate 153 -> 143
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 143.seg.nrrd --prediction "${predictions}RT 153 143-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 143" --prefix_pred "RT_153_143"

# evaluate 153 -> 144
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 144.seg.nrrd --prediction "${predictions}RT 153 144-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 144" --prefix_pred "RT_153_144"

# # evaluate 153 -> 145 
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 145.seg.nrrd --prediction "${predictions}RT 153 145-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 145" --prefix_pred "RT_153_145"

# evaluate 153 -> 146 
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 146.seg.nrrd --prediction "${predictions}RT 153 146-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 146" --prefix_pred "RT_153_146"

# evaluate 153 -> 147 
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 147.seg.nrrd --prediction "${predictions}RT 153 147-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 147" --prefix_pred "RT_153_147"

# # evaluate 153 -> 150
# python3 eval.py --ground_truth ${base}Segmentation\ RT\ 150.seg.nrrd --prediction "${predictions}RT 153 150-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 150" --prefix_pred "RT_153_150"

# evaluate 153 -> 152
python3 eval.py --ground_truth ${base}Segmentation\ RT\ 152.seg.nrrd --prediction "${predictions}RT 153 152-syn80-demons.seg.nrrd" --dice --eval_indices 1 2 3 4 9 10 --mesh_cache ssm_out --prefix_truth "RT 152" --prefix_pred "RT_153_152"