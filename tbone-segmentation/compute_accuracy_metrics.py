"""
compute_accuracy_metrics.py

Compute dice scores and Hausdorff distances for segment ids

"""
import os 
import sys 
import argparse 
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
import surface_distance
from surface_distance.metrics import *
from utils.file_io import *


def parse_command_line(args):
	'''

	'''

	print('parsing command line')

	parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
	parser.add_argument('--template', 
						action="store", 
						type=str, 
						default="153"
						)
	parser.add_argument('--side', 
						action="store", 
						type=str, 
						default="RT"
						)
	parser.add_argument('--base', 
						action="store", 
						type=str, 
						default="/Volumes/Extreme SSD/ANTs-registration"
						)
	parser.add_argument('--target',
						action="store",
						type=str,
						nargs='*',
						help="specify which scan should be the target scan. \
						If a scan is specified, only the registration of the specified target scan onto the template scan will be performed. \
						If the target scan is not specified, the entire list of scans will be registered onto the template." 
					   	)
	parser.add_argument('--dry',
						action="store_true"
						)
	parser.add_argument('--downsample',
						action="store_true",
						help="downsamples the images for faster registration"
                    	)
	parser.add_argument('--downsample_size',
						action="store",
						type=int,
						default=300
                    	)
	parser.add_argument('--flip',
						action="store_true",
						help="determines if the side the target scan is different from the template")
	parser.add_argument('--ids',
						type=int,
						nargs='+',
						help="segment indices (1-indexed) to calculate accuracy metrics")
	
	args = vars(parser.parse_args())
	return args

def main(): 
	args = parse_command_line(sys.argv)
	side = args['side']
	base = args['base']
	template = args['template']
	target = args['target']
	ids = args['ids']
	gt_dir = os.path.join(base, 'segmentations')
	gt_dir_nii = os.path.join(base, 'segmentations', 'NIFTI Segmentations')
	pred_dir = os.path.join(base, 'predictions', 'NIFTI Predictions')
	RT = [	'138', 
			'142',
			'143', 
			'144',
			'146', 
			'147',
			'152', 
			'153'
		]
	LT = [	'138',
			'143',
			'144',
			'145', 
			'146', 
			'148',
			'151',
			'171'
			]
	if side == 'RT':
		other_side = 'LT'
		scan_id = RT
		opposite_scan_id = LT
	else: 
		other_side = 'RT'
		scan_id = LT
		opposite_scan_id = RT
	
	if args['flip']: target_scan_id = opposite_scan_id
	else: target_scan_id = scan_id	
	
	if args['target'] is not None:
		for target in args['target']:
			if target not in target_scan_id: 
				print('incorrectly specified target scan')
				return
		target_scan_id = args['target']

	template_seg_path = adjust_file_path(gt_dir, "Segmentation %s %s"%(side, template), ".seg.nrrd", registration=None)
	seg_names = get_segmentation_names(nrrd.read_header(template_seg_path))

	# Initialize metric dictionaries
	dice_dict, hausdorff_dict = dict(), dict()
	dice_dict['Target'], hausdorff_dict['Target'] = [], []
	for i in ids: # seg_names are 0-indexed, while ids are 1-indexed (due to presence of background class in one-hot)
		dice_dict[seg_names[i-1]], hausdorff_dict[seg_names[i-1]] = [], []
		
	for target in target_scan_id:
		if template in target: 
			continue
		else:
			print('-- Evaluating %s'%(target))
			pred_seg_path = adjust_file_path(pred_dir, "Segmentation %s %s %s"%(side, template, target), ".nii.gz", args['downsample'], args['downsample_size'], flip=args['flip'])
			
			if args['flip']:
				gt_seg_path_nrrd = adjust_file_path(gt_dir, "Segmentation %s %s"%(other_side, target), ".seg.nrrd", registration=None)
				gt_seg_path_nii = adjust_file_path(gt_dir_nii, "Segmentation %s %s"%(other_side, target), ".nii.gz", registration=None)
			else:
				gt_seg_path_nrrd = adjust_file_path(gt_dir, "Segmentation %s %s"%(side, target), ".seg.nrrd", registration=None)
				gt_seg_path_nii = adjust_file_path(gt_dir_nii, "Segmentation %s %s"%(side, target), ".nii.gz", registration=None)

			pred_seg = np.array(nib.load(pred_seg_path).dataobj)
			gt_seg = np.array(nib.load(gt_seg_path_nii).dataobj)
			spacing = np.linalg.norm(nrrd.read_header(gt_seg_path_nrrd)['space directions'][1:], axis=0)

			pred_one_hot = np.moveaxis((np.arange(pred_seg.max()+1) == pred_seg[...,None]), -1, 0)
			gt_one_hot = np.moveaxis((np.arange(gt_seg.max()+1) == gt_seg[...,None]), -1, 0)

			print(pred_one_hot.shape)
			print(gt_one_hot.shape)

			if args['flip']: target_name = other_side + ' ' + target
			else: target_name = side + ' ' + target

			dice_dict['Target'].append(target_name)
			hausdorff_dict['Target'].append(target_name)

			for i in ids:
				seg_name = seg_names[i-1]
				print('---- Computing metrics for segment: %s'%(seg_name))
				dice_coeff = compute_dice_coefficient(gt_one_hot[i], pred_one_hot[i])
				surface_distances = compute_surface_distances(gt_one_hot[i], pred_one_hot[i], spacing)
				mod_hausdorff_distance = max(compute_average_surface_distance(surface_distances))

				dice_dict[seg_name].append(dice_coeff)
				hausdorff_dict[seg_name].append(mod_hausdorff_distance)
	
	dice_df = pd.DataFrame.from_dict(dice_dict)
	hausdorff_df = pd.DataFrame.from_dict(hausdorff_dict)

	print('Dice Scores')
	print(dice_df)

	print('Modified Hausdorff Distances')
	print(hausdorff_df)

	if args['flip']:
		dice_path = os.path.join(base, 'dice ' + '-'.join(str(i) for i in ids) + '-flipped.csv')
		hausdorff_path = os.path.join(base, 'hausdorff ' + '-'.join(str(i) for i in ids) + '-flipped.csv')
	else:
		dice_path = os.path.join(base, 'dice ' + '-'.join(str(i) for i in ids) + '.csv')
		hausdorff_path = os.path.join(base, 'hausdorff ' + '-'.join(str(i) for i in ids) + '.csv')

	dice_df.to_csv(dice_path)
	hausdorff_df.to_csv(hausdorff_path)

	return 

if __name__ == "__main__":
	main()
