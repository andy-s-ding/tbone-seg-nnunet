"""ssm.py


yao deformabel template match
"""
import numpy
from pycpd import DeformableRegistration, RigidRegistration
import nrrd
import os 
import sys 
import argparse

from utils.file_io import *
from utils.mesh_ops import *
from utils.cpd_wrapper import *

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
	parser.add_argument("--build", 
						action="store_true",
						help="calculate the PCA decomposition"
						)
	parser.add_argument("--mesh", 
						action="store_true",
						help="extract surface meshes from segmentation nrrd files"
						)
	parser.add_argument("--eval_indices", "-i",
						nargs="+",
						metavar="idx",
						type=int,
						default=1,
						help="specify the indices from the seg nrrd that should be included in analysis.",
						)
	parser.add_argument("--output_dir", "-o",
						action="store",
						type=str,
						default="ssm_out",
						help="specify location meshes and ssm information should be written",
						)
	parser.add_argument("--dry",
						action="store_true",
						help="if specified, calculations will be skipped. for debugging use. "
					   )

	args = vars(parser.parse_args())
	return args


def extract_meshes(segmentation_nrrds, segment_indices, side, segmentation_dir, output_dir, dry=True): 

	segment_names = []
	# loop over specified segmentations
	for i, nrrd_num in enumerate(segmentation_nrrds): 
		print(nrrd_num)
		# compile the path to the segmentation
		path_seg_nrrd = os.path.join(segmentation_dir, "Segmentation %s %s.seg.nrrd" % (side, nrrd_num))
		# path_seg_nrrd = os.path.join(segmentation_dir, "%s 153 %s-syn80-demons-downsample300.seg.nrrd" % (side, nrrd_num))

		# read in segmentation
		volume, header = nrrd.read(path_seg_nrrd)
		volume = convert_to_one_hot(volume, header)

		# print(volume.shape)

		# extract segmentation names
		index_names = get_segmentation_names(header, indices=segment_indices)

		if i == 0: 
			segment_names = index_names

		name_consistency_check(segment_names, index_names)

		# loop over specified indices
		for idx, name in zip(segment_indices, index_names): 

			if dry: 
				print(nrrd_num, idx, name)
				continue

			# extract meshes for each index 
			surface_mesh = obtain_surface_mesh(volume, header, idx=idx)

			# save mesh
			surface_mesh.save(os.path.join(output_dir, "%s %s %s mesh.vtk") % (side, nrrd_num, name))
			# surface_mesh.save(os.path.join(output_dir, "%s_153_%s %s mesh.vtk") % (side, nrrd_num, name), binary=False)

	return {segment_idx: segment_names[i] for i, segment_idx in enumerate(segment_indices)}



def compute_ssm(all_mesh_points, target_idx=0, restrict_source_points=800, restrict_target_points=800): 

	target = all_mesh_points[0]
	target_indices = np.random.randint(0, target.shape[0]-1, size=restrict_target_points)
	target = target[target_indices]
	# register n-1 mesh -> 1 mesh with CPD 

	transformed_targets = [target]
	for i, points_source in enumerate(all_mesh_points): 
		if i == target_idx: 
			continue

		# downsample source
		source_indices = np.random.randint(0, points_source.shape[0]-1, size=restrict_source_points)
		points_source = points_source[source_indices]

		# initial rigid registration
		transformed_target, _ = initial_rigid_registration(target, points_source, visualize_reg=True)

		# deformable registration
		transformed_target, _ = deformable_registration(transformed_target, points_source, visualize_reg=True)

		# add to list 
		transformed_targets.append(transformed_target)

	for i in transformed_targets[1:]: 
		continue


	# PCA 

	return 


def build_ssm(segmentation_nrrds, index_name, side, output_dir, dry=True):

	all_mesh_points = []
	# read in mesh 
	for nrrd_num in segmentation_nrrds:
		mesh_path = os.path.join(output_dir, "%s %s %s mesh.vtk") % (side, nrrd_num, index_name)
		mesh = read_vtk(mesh_path)
		points = return_surface_points(mesh)

		print(points.shape)

		all_mesh_points.append(points)

	if dry: 
		return
	compute_ssm(all_mesh_points)

	return


def main(): 
	args = parse_command_line(sys.argv)
	print(args)

	if args['all']: 
		args["images"] = [
			# '138',
			'142',
			# '142-average',
			'143',
			'144',
			# '145',
			'146',
			'147',
			# '150',
			'152',
			# '153'
		]

	if isinstance(args["eval_indices"], int): 
		args["eval_indices"] = [args["eval_indices"]]

	segmentation_dir = os.path.join(args["base"], "segmentations")
	predictions_dir = os.path.join(args["base"], "predictions")

	# extract meshes
	if args["mesh"]: 
		segment_names = extract_meshes(args["images"], args["eval_indices"], 
									   args["side"], segmentation_dir, args["output_dir"], dry=args["dry"])
		# write_to_file('segment_names', segment_names)
	else: 
		segment_names = read_from_file('segment_names')

	# build ssm
	if args["build"]: 
		for idx in args["eval_indices"]: 
			index_name = segment_names[idx]
			build_ssm(args["images"], index_name, args["side"], args["output_dir"], dry=args["dry"])


	return


if __name__ == "__main__": 

	main()
