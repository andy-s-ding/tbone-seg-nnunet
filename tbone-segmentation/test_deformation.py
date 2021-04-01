"""
test_deformation.py

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

from utils.file_io import *
from utils.mask import *


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
	parser.add_argument('--cached',
						action="store",
						type=str, 
						default="/Volumes/Extreme SSD/ANTs-registration/ssm_transforms/"
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
	parser.add_argument('--num_deforms',
						action="store",
						type=int,
						default=100
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

		print("applying transformation to channel %d" % i)

		print(segment.components)
		print(segment.pixeltype)
		print(segment.shape)

		predicted_target = ants.apply_transforms(
			target_image, segment, transform, interpolator=interpolator, verbose=True)
		predicted_targets.append(predicted_target)

	predicted_targets_image = ants.merge_channels(predicted_targets)

	return predicted_targets_image

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


def deform(template, base, side, save_dir, transform, id, downsample=False, downsample_size=300, write_segmentations=True):
	if side == 'RT':
		other_side = 'LT'
	else:
		other_side = 'RT'

	template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")
	template_image = ants.image_read(template_path)
	template_ants_header = ants.image_header_info(template_path)
	template_image_header = nrrd.read_header(template_path)

	deformed_volume = apply_transform_to_image(transform, template_image, template_image, interpolator='linear')
	deformed_volume_path_nrrd = adjust_file_path(save_dir, "%s %s deform%d" % (side, template, id), ".nrrd", registration=None, downsample=downsample, downsample_size=downsample_size)
	deformed_volume_path_nii = adjust_file_path(save_dir, "%s %s deform%d" % (side, template, id), ".nii.gz", registration=None, downsample=downsample, downsample_size=downsample_size)
	ants_image_to_file(deformed_volume, template_image_header, template_image_header, deformed_volume_path_nrrd, segmentations=False)
	# ants_image_to_file(deformed_volume, template_image_header, template_image_header, deformed_volume_path_nii, segmentations=False, nifti=True)

	if write_segmentations:
		template_segmentation_path = os.path.join(base, "segmentations/Segmentation %s %s.seg.nrrd" % (side, template))
		template_segmentations = ants.image_read(template_segmentation_path, pixeltype="unsigned char")
		template_segmentations_header = nrrd.read_header(template_segmentation_path)

		predicted_segmentations = apply_transform_to_image(transform, template_image, template_segmentations)
		predicted_targets_path_nrrd = adjust_file_path(save_dir, "Segmentation %s %s deform%d" % (side, template, id), ".seg.nrrd", registration=None, downsample=downsample, downsample_size=downsample_size)
		predicted_targets_path_nii = adjust_file_path(save_dir, "Segmentation %s %s deform%d" % (side, template, id), ".nii.gz", registration=None, downsample=downsample, downsample_size=downsample_size)
		ants_image_to_file(predicted_segmentations, template_segmentations_header, template_image_header, predicted_targets_path_nrrd)
		# ants_image_to_file(predicted_segmentations, template_segmentations_header, template_image_header, predicted_targets_path_nii, nifti=True, ants_header=template_ants_header)

	return

def main(): 
	args = parse_command_line(sys.argv)
	side = args['side']
	base = args['base']
	template = args['template']
	pred_dir = os.path.join(base, 'predictions')
	save_dir = os.path.join(pred_dir, 'NIFTI Predictions')
	num_deforms = args['num_deforms']
	ids = range(1,num_deforms+1)

	# for deform_id in ids:
	# 	deform_path = adjust_file_path(args["cached"], "%s %s deform%d" % (side, template, deform_id), ".nii.gz", registration='inverse',
	# 											downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
	# 	deformed_volume_path = adjust_file_path(save_dir, "%s %s deform%d" % (side, template, deform_id), ".nrrd", registration=None,
	# 											downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])

	# 	if args['overwrite'] or not(os.path.exists(deformed_volume_path)):
	# 		deform(template, base, side, save_dir, [deform_path], deform_id, downsample=args['downsample'], downsample_size=args['downsample_size'])
		
	# 	else:
	# 		print('-- deformed volume already exists at %s' % (deformed_volume_path))
	# return 

	for deform_id in ids:
		deformed_seg_path = adjust_file_path(pred_dir, "Segmentation %s %s deform%d" % (side, template, deform_id), ".seg.nrrd", registration=None,
												downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])

		deformed_nii_path = adjust_file_path(save_dir, "Segmentation %s %s deform%d" % (side, template, deform_id), ".nii.gz", registration=None,
											downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
		
		if args['overwrite'] or not(os.path.exists(deformed_nii_path)):
			ants_img = ants.image_read(deformed_seg_path)
			header = nrrd.read_header(deformed_seg_path)
			data = ants_img.view(single_components=True)
			data = convert_to_one_hot(data, header)
			fg = np.max(data, axis=0)
			labelmap = np.multiply(np.argmax(data, axis=0) + 1, fg).astype('uint8')
			deformed_seg = ants.from_numpy(labelmap, origin=ants_img.origin, spacing = ants_img.spacing, direction=ants_img.direction)
			deformed_seg.to_filename(deformed_nii_path)
		else:
			print('-- deformed segmentation already exists at %s' % (deformed_nii_path))

if __name__ == "__main__":
	main()
