"""
generate_ssm_deformed_data.py

Deform a template and its labels based on SSM-generated deformation fields

"""
import os 
import sys 
import argparse 
import numpy as np
import ants
import nrrd

from utils.file_io import *
from utils.mask import *
from utils.transforms import *


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
						default="/media/andyding/EXTREME SSD/ANTs-registration/"
						)
	parser.add_argument('--cached',
						action="store",
						type=str, 
						default="/media/andyding/EXTREME SSD/ANTs-registration/ssm_transforms/"
						)
	parser.add_argument('--downsample',
						action="store_true",
						help="downsamples the images for faster registration"
						)
	parser.add_argument('--downsample_size',
						action="store",
						type=int,
						default=100
						)
	parser.add_argument('--num_deforms',
						action="store",
						type=int,
						default=100
						)
	parser.add_argument('--start_id',
						action="store",
						type=int,
						default=1
						)
	parser.add_argument('--overwrite',
					 	action="store_true",
						help="determines whether to overwrite existing output if present"
					 	)
	parser.add_argument('--segs',
					 	action="store_true",
						help="determines whether to write segmentation files"
					 	)
	parser.add_argument('--nifti',
					 	action="store_true",
						help="determines whether to write nifti files in addition to nrrd files"
					 	)
	parser.add_argument('--flip',
						action="store_true",
						help="determines if the side the target scan is different from the template")
	
	args = vars(parser.parse_args())
	return args


def deform(template, base, side, transform, id, downsample=False, downsample_size=300, write_nifti=False, write_segmentations=False):
	if side == 'RT':
		other_side = 'LT'
	else:
		other_side = 'RT'

	template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")
	template_image = ants.image_read(template_path)
	template_ants_header = ants.image_header_info(template_path)
	template_image_header = nrrd.read_header(template_path)

	deformed_volume = apply_transform_to_image(
		transform, template_image, template_image, interpolator='linear')
	deformed_volume_path_nrrd = adjust_file_path(deformed_volumes_dir, "%s %s deform%d" % (
		side, template, id), ".nrrd", registration=None, downsample=downsample, downsample_size=downsample_size)
	deformed_volume_path_nii = adjust_file_path(deformed_volumes_nii_dir, "%s %s deform%d" % (
		side, template, id), ".nii.gz", registration=None, downsample=downsample, downsample_size=downsample_size)
	ants_image_to_file(deformed_volume, template_image_header,
	                   template_image_header, deformed_volume_path_nrrd, segmentations=False)
	if write_nifti:
		ants_image_to_file(deformed_volume, template_image_header, template_image_header,
		                   deformed_volume_path_nii, segmentations=False, nifti=True)

	if write_segmentations:
		template_segmentation_path = os.path.join(
			base, "segmentations/Segmentation %s %s.seg.nrrd" % (side, template))
		template_segmentations = ants.image_read(
			template_segmentation_path, pixeltype="unsigned char")
		template_segmentations_header = nrrd.read_header(template_segmentation_path)

		predicted_segmentations = apply_transform_to_image(
			transform, template_image, template_segmentations)
		predicted_targets_path_nrrd = adjust_file_path(pred_dir, "Segmentation %s %s deform%d" % (
			side, template, id), ".seg.nrrd", registration=None, downsample=downsample, downsample_size=downsample_size)
		predicted_targets_path_nii = adjust_file_path(pred_nii_dir, "Segmentation %s %s deform%d" % (
			side, template, id), ".nii.gz", registration=None, downsample=downsample, downsample_size=downsample_size)
		ants_image_to_file(predicted_segmentations, template_segmentations_header,
		                   template_image_header, predicted_targets_path_nrrd)
		if write_nifti:
			ants_image_to_file(predicted_segmentations, template_segmentations_header,
			                   template_image_header, predicted_targets_path_nii, nifti=True)
	return

def main(): 
	args = parse_command_line(sys.argv)
	side = args['side']
	base = args['base']
	template = args['template']
	global pred_dir
	global pred_nii_dir
	global deformed_volumes_dir
	global deformed_volumes_nii_dir
	pred_dir = os.path.join(base, 'predictions')
	pred_nii_dir = os.path.join(pred_dir, 'NIFTI Predictions')
	deformed_volumes_dir = os.path.join(base, 'deformed_volumes')
	deformed_volumes_nii_dir = os.path.join(deformed_volumes_dir, 'NIFTI Volumes')
	start_id = args['start_id']
	num_deforms = args['num_deforms']
	ids = range(start_id,start_id+num_deforms)

	for deform_id in ids:
		deform_path = adjust_file_path(args["cached"], "%s %s deform%d" % (side, template, deform_id), ".nii.gz", registration='inverse',
												downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
		deformed_volume_path_nrrd = adjust_file_path(deformed_volumes_dir, "%s %s deform%d" % (side, template, deform_id), ".nrrd", registration=None,
												downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])

		if args['overwrite'] or not(os.path.exists(deformed_volume_path_nrrd)):
			deform(template, base, side, [deform_path], deform_id, downsample=args['downsample'], downsample_size=args['downsample_size'],
					write_nifti=args['nifti'], write_segmentations=args['segs'])
		else:
			print('-- deformed volume already exists at %s' % (deformed_volume_path_nrrd))
	return 

if __name__ == "__main__":
	main()
