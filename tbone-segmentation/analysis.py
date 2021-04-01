"""analysis.py


yao deformabel template match
"""
import numpy
import nrrd
import os 
import sys 
import argparse

from utils.file_io import *
from utils.mesh_ops import *
from utils.metrics import *


def parse_command_line(args):

	print("parsing command line")

	parser = argparse.ArgumentParser(description="Registration pipeline for image-image registration")
	parser.add_argument("--base", 
						action="store", 
						type=str, 
						default="/scratch/groups/rtaylor2/ANTs-registration"
						)
	parser.add_argument("--side", 
						action="store", 
						type=str, 
						default="RT", 
						help="specify the anatomical side. used to identify paths, file names"
						)
	parser.add_argument("--images", 
						nargs="+", 
						metavar="id",
						type=str, 
						default="144",
						help="specify the scan IDs to be included in the analysis. used to identify paths, file names"
						)
	parser.add_argument("--all", 
						action="store_true",
						help="if specified, all scan IDs will be included in the analysis." 
						)
	parser.add_argument("--volumes", 
						action="store_true",
						help="if specified, volumes for associated meshes will be calculated." 
						)
	parser.add_argument("--eval_indices", "-i",
						nargs="+",
						metavar="idx",
						type=int,
						default=1,
						help="specify the indices from the seg nrrd that should be included in analysis.",
						)
	parser.add_argument("--dry",
						action="store_true",
						help="if specified, calculations will be skipped. for debugging use. "
					   )

	args = vars(parser.parse_args())
	return args


def main(): 
	args = parse_command_line(sys.argv)
	print(args)

	if args['all']: 
		args["images"] = [
			'138',
			'142',
			'143',
			'144',
			'146',
			'147',
			'152',
			# '153'
		]

	if isinstance(args["eval_indices"], int): 
		args["eval_indices"] = [args["eval_indices"]]

	segmentation_dir = os.path.join(args["base"], "segmentations")
	prediction_dir = os.path.join(args["base"], "predictions")

	all_meshes = {nrrd_num: {} for nrrd_num in args["images"]}
	all_meshes['153'] = {}

	# output_dict = dict()
	# output_dict['segment_id'] = []
	# output_dict['volume'] = []
	# for idx in args["eval_indices"]:
	# 	output_dict["segment_{}".format(idx)] = []

	for nrrd_num in args["images"]: 
		# path_seg_nrrd = os.path.join(segmentation_dir, "Segmentation %s %s.seg.nrrd" % (args["side"], nrrd_num))
		path_seg_nrrd = os.path.join(prediction_dir, "%s 153 %s-syn80-demons.seg.nrrd" % (args["side"], nrrd_num))
		segment_names = get_segmentation_names(nrrd.read_header(path_seg_nrrd), indices=args["eval_indices"])
		print(segment_names)
		
		for i, idx in enumerate(args["eval_indices"]):
			# print("segment {}".format(idx))
			index_name = segment_names[i]

			# read in mesh 
			# mesh_path = os.path.join("ssm_out", "%s %s %s mesh.vtk") % (args["side"], nrrd_num, index_name)
			mesh_path = os.path.join("ssm_out", "%s_153_%s %s mesh.vtk") % (args["side"], nrrd_num, index_name)
			mesh = read_vtk(mesh_path)
			all_meshes[nrrd_num][index_name] = mesh

		# calculate volumes
		if args["volumes"]: 
			volumes = calc_volume(all_meshes)
			# print(volumes)
			# print_scores(volumes)

		# calculate landmark distances
		# if args["landmarks"]: 

			# pass
			# extract landmarks - should be dict of {landmark: points}

			# call specific functions for calculation between pairs of landmarks

			# pass landmarks, meshes in to calculate the distances
			# landmark_dists = calculate_landmark_distances(all_meshes)

			# output results
			# print_scores(landmark_dists)
	if args["volumes"]:
		
		# include template volumes
		path_seg_nrrd = os.path.join(segmentation_dir, "Segmentation %s 153.seg.nrrd" % (args["side"]))
		segment_names = get_segmentation_names(nrrd.read_header(path_seg_nrrd), indices=args["eval_indices"])
		
		for i, idx in enumerate(args["eval_indices"]):
			index_name = segment_names[i]

			# read in mesh 
			mesh_path = os.path.join("ssm_out", "%s 153 %s mesh.vtk") % (args["side"], index_name)
			mesh = read_vtk(mesh_path)
			all_meshes['153'][index_name] = mesh

		volumes = calc_volume(all_meshes)

		# write dictionary of volumes
		output_dict = dict()
		output_dict['scan_id'] = []
		for segment in segment_names:
			output_dict[segment] = []

		for scan_id, scan_volumes in volumes.items():
			output_dict['scan_id'].append(scan_id)
			for segment in segment_names:
				output_dict[segment].append(scan_volumes[segment])
	
	print(output_dict)
	write_to_file('RT 153 Propagated Volumes', output_dict)

	return


if __name__ == "__main__": 

	main()
