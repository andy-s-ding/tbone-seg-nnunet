"""
deform_volumes.py

Andy's copy of registration_pipeline with functionality for volume deformation

"""
import os 
import sys 
import argparse 
import numpy as np
import ants
import nrrd

import shutil
import psutil
import gc
import time

from utils.file_io import ants_image_to_file
from utils.mask import flip_image


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
						default="/Volumes/Extreme SSD/ANTs-registration/"
						)
	parser.add_argument('--target',
						action="store",
						type=str,
						help="specify which scan should be the target scan. \
						If a scan is specified, only the registration of the specified target scan onto the template scan will be performed. \
						If the target scan is not specified, the entire list of scans will be registered onto the template." 
					   	)
	parser.add_argument('--cached',
						action="store",
						type=str, 
						default="/Volumes/Extreme SSD/ANTs-registration/transforms/"
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
	parser.add_argument('--overwrite',
					 	action="store_true",
						help="determines whether to overwrite existing output .nrrds if present"
					 	)
	parser.add_argument('--flip',
						action="store_true",
						help="determines if the side the target scan is different from the template")
	
	args = vars(parser.parse_args())
	return args


def adjust_file_path(save_dir, prefix, suffix, downsample=None, downsample_size=300, registration="syn80-demons", is_annotation=False, flip=False): 

	path = os.path.join(save_dir, prefix)

	if registration:
		path += "-" + registration

	if downsample: 
		path += "-downsample%d"%(downsample_size)

	if flip:
		path += "-flipped"

	if is_annotation: 
		path += "-annotations"

	path += suffix

	print(" -- returning path: %s" % path)

	return path


def apply_transform_to_image(transform, target_image, template_image, interpolator='genericLabel'): 
	"""apply_transform_to_image

	Apply transformations to an image. The image could either be a multichannel segmentation (0, 1) in each channel
	or a single channel segmentation (0: num indices)

	Transform should be in the order of an ants fwd transform, i.e. [deformation path, affine path]
	
	Args:
		transform (list): list of file paths to transformations to be applied 
		target_image (ANTsImage): the image with spatial metadata to be applied after transformation
		template_image (ANTsImage): the image that will be transformed
		interpolator (str, optional): the interpolation method
	
	Returns:
		TYPE: Description
	"""

	# directly apply the transformation if the channels do not have to be split
	if template_image.components == 1: 
		return ants.apply_transforms(target_image, template_image, transform, interpolator=interpolator, verbose=True)

	# split the channels and apply transformation to each individually 
	individual_segments = ants.split_channels(template_image)
	print(len(individual_segments))
	predicted_targets = []
	for i, segment in enumerate(individual_segments): 

		print("applying transformation to channel %d"%i)

		print(segment.components)
		print(segment.pixeltype)
		print(segment.shape)

		predicted_target = ants.apply_transforms(target_image, segment, transform, interpolator=interpolator, verbose=True)
		predicted_targets.append(predicted_target)

	predicted_targets_image = ants.merge_channels(predicted_targets)

	return predicted_targets_image


def deform(template, target, base, side, save_dir, transform, downsample=False, downsample_size=300, flip=False):  
	if side == 'RT': other_side = 'LT'
	else: other_side = 'RT'

	template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")
	template_image = ants.image_read(template_path)
	template_image_header = nrrd.read_header(template_path)

	if flip:
		target_path = os.path.join(base, "images", other_side + ' ' + target + ".nrrd")
		template_image = flip_image(template_image)
	else: target_path = os.path.join(base, "images", side + ' ' + target + ".nrrd")

	target_image = ants.image_read(target_path)
	target_image_header = nrrd.read_header(target_path)

	deformed_volume = apply_transform_to_image(transform, target_image, template_image, interpolator='linear')
	deformed_volume_path = adjust_file_path(save_dir, "%s %s %s"%(side, template, target), ".nrrd", downsample=downsample, downsample_size=downsample_size, flip=flip)
	ants_image_to_file(deformed_volume, template_image_header, target_image_header, deformed_volume_path, segmentations=False)
	

def main(): 
	args = parse_command_line(sys.argv)

	side = args['side']
	base = args['base']
	template = args['template']
	target = args['target']
	save_dir = os.path.join(base, 'deformed_volumes')
	RT = [	'138', 
			'142',
			'143', 
			'144',
			'146', 
			'147',
			'152', 
			'153',
			'170',
			'174', 
			'175',
			'177',
			'179',
			'183',
			'189',
			'191',
			'192',
			'195'
		]
	LT = [	'138',
			'143',
			'144',
			'145', 
			'146', 
			'147',
			'148',
			'151', 
			'152',
			'169', 
			'170', 
			'171', 
			'172',
			'173',
			'175',
			'176',
			'177',
			'183',
			'185',
			'191',
			'192',
			'193',
			'195'
			]
	if side == 'RT':
		scan_id = RT
		opposite_scan_id = LT
	else: 
		scan_id = LT
		opposite_scan_id = RT
	
	if args['flip']: target_scan_id = opposite_scan_id
	else: target_scan_id = scan_id
	
	if args['target'] is not None: 
		if args['target'] not in target_scan_id: 
			print('incorrectly specified target scan')
			return
		target_scan_id = [args['target']]

	# for target in target_scan_id:
	# 	if template in target: 
	# 		continue
	# 	else:
	# 		deformed_volume_path = adjust_file_path(save_dir, "%s %s %s"%(side, template, target), ".nrrd", args['downsample'], args['downsample_size'], flip=args['flip'])
	# 		deformed_nii_path = adjust_file_path(save_dir, "%s %s %s"%(side, template, target), ".nii.gz", downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
	# 		if args['overwrite'] or not (os.path.exists(deformed_nii_path)):
	# 			deformed_volume = ants.image_read(deformed_volume_path)
	# 			print('-- Saving NII')
	# 			deformed_volume.to_filename(deformed_nii_path)
	# 		else:
	# 			print ('-- Nifti file already exists at %s' % deformed_volume_path)
	
	for id in range(1,101):
		deformed_nii_path = os.path.join(save_dir, '%s %s deform%d-downsample%d.nii.gz' % (side, template, id, args['downsample_size']))
		deformed_volume_path = os.path.join(save_dir, '%s %s deform%d-downsample%d.nrrd' % (side, template, id, args['downsample_size']))
		if args['overwrite'] or not (os.path.exists(deformed_nii_path)):
			deformed_volume = ants.image_read(deformed_volume_path)
			print('-- Saving NII')
			deformed_volume.to_filename(deformed_nii_path)
		else:
			print('-- Nifti file already exists at %s' % deformed_nii_path)
	return 


if __name__ == "__main__":
	main()
