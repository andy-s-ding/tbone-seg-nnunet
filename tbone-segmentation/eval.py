"""
eval.py

"""
import numpy as np
import os 
import sys 
import nrrd
import argparse

from utils.metrics import calc_dice
from utils.metrics import calc_hausdorff, calc_volume_dice
from utils.file_io import print_scores, read_from_file, convert_to_one_hot


def parse_command_line(args):
	'''

	'''

	print('parsing command line')

	parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
	parser.add_argument('--ground_truth',
						action="store", 
						type=str, 
						default="142"
						)
	parser.add_argument('--prediction', 
						action="store", 
						type=str, 
						default="/scratch/groups/rtaylor2/ANTs-registration"
						)
	parser.add_argument('--out', 
						action="store", 
						type=str, 
						default="ssm_out"
						)
	parser.add_argument('--dice',
						action="store_true"
					   )
	parser.add_argument('--volume_dice',
						action="store_true"
					   )
	parser.add_argument('--hausdorff',
						action="store_true"
					   )
	parser.add_argument('--eval_indices',
						nargs='+',
						)
	parser.add_argument('--intermediate',
						action="store_true",
						)
	parser.add_argument('--mesh_cache', 
						action="store",
						type=str,
						)
	parser.add_argument('--prefix_truth',
						action="store",
						type=str,
						)
	parser.add_argument('--prefix_pred', 
						action="store",
						type=str,
						)
	
	args = vars(parser.parse_args())
	return args


def main(): 
	args = parse_command_line(sys.argv)
	print(args)

	segmentation_truth, header_truth = nrrd.read(args['ground_truth'])
	segmentation_pred, header_pred = nrrd.read(args['prediction'])

	segmentation_truth = convert_to_one_hot(segmentation_truth, header_truth)
	segmentation_pred = convert_to_one_hot(segmentation_pred, header_pred)

	print(segmentation_truth.shape)
	print(segmentation_pred.shape)

	print(args['eval_indices'])

	if args['dice']: 
		print('dice')
		scores_dice = calc_dice(segmentation_truth, segmentation_pred, indices=args['eval_indices'])
		print_scores(scores_dice, names=args['eval_indices'])

	if args['volume_dice']: 
		print('volume-based dice calculation')
		dices, _, _ = calc_volume_dice(segmentation_truth, header_truth, 
										segmentation_pred, header_pred, 
										indices = args['eval_indices'],
										mesh_cache = args['mesh_cache'],
										prefix_truth = args['prefix_truth'],
										prefix_pred = args['prefix_pred']
										)

		segment_name = read_from_file("segment_names")
		print_scores(dices, names=[segment_names[int(idx)] for idx in args['eval_indices']])

	if args['hausdorff']: 
		print('hausdorff')
		scores_hausdorff, _, pred_meshes= calc_hausdorff(segmentation_truth, header_truth, 
														segmentation_pred, header_pred, 
														indices = args['eval_indices'],
														mesh_cache = args['mesh_cache'],
														prefix_truth = args['prefix_truth'],
														prefix_pred = args['prefix_pred']
														)
		segment_names = read_from_file("segment_names")

		# store the prediction meshes (assumes that the gt meshes have already been stored)
		if args['intermediate'] is not None: 
			for idx, mesh in zip(args['eval_indices'], pred_meshes): 
				mesh_path = os.path.join(args["out"], "%s %s mesh.vtk" % (args["prefix_pred"], segment_names[int(idx)]))
				mesh.save(mesh_path)
		
		print_scores(scores_hausdorff, names=[segment_names[int(idx)] for idx in args['eval_indices']])


	return

if __name__ == "__main__": 
	main()