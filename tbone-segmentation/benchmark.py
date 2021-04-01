"""
benchmark.py


The call signature of the register to target function has been modified relative to registration_pipeline.py 
to facilitate the direct registration of one image onto another, and the propagation of associated segmentations 

This design makes it easier to benchmark our templates against the NIH OpenEar library. 

the order of the segments in DELTA is: 
- scala tympani
- scala_vestibuli
- malleus
- incus 
- stapes
- cochleovestibular nerve
- facial nerve
- chorda tympani
- tympanic membrane
- outer ear canal
- carotis interna
- sinus dura
- bone


"""
import os
import sys
import argparse
import numpy as np
import ants
import nrrd
import gc

from utils.file_io import ants_image_to_file
from utils.mask import flip_image, invert_mask, create_mask

def parse_command_line(args):
	'''

	'''

	print('parsing command line')

	parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
	parser.add_argument('--template', 
						action="store", 
						type=str, 
						)
	parser.add_argument('--template_seg',
						action="store",
						type=str,
						)
	parser.add_argument('--base', 
						action="store", 
						type=str, 
						default="/scratch/groups/rtaylor2/ANTs-registration"
						)
	parser.add_argument('--target',
						action="store",
						type=str,
						help="specify which scan should be the target." 
					   )
	parser.add_argument('--target_mask',
						action="store",
						type=str,
						help="specify the mask which should be used for the target image" 
					   )
	parser.add_argument('--initial_transform',
						action="store",
						type=str,
						help="specify the initial transformation for the image")
	parser.add_argument('--dry',
						action="store_true"
						)
	parser.add_argument('--flip',
						action="store_true"
						)
	parser.add_argument('--downsample',
						action="store_true"
					   )
	parser.add_argument('--output_prefix',
						action="store",
						type=str,
						default="RT_153_"
						)
	parser.add_argument('--target_scan_id', 
						action="store",
						type=str,
						)
	args = vars(parser.parse_args())
	return args


def register_to_target(template_path, template_segmentation_path, target_path, base, save_dir, prefix, targ_id, 
		target_mask_path=None, initial_transform=None, flip=False, dry=False, downsample=False, iteration=None, downsample_size=300):
	'''
	
	'''

	print("---"*10)
	print("entering registration w/ template %s and target %s" % (template_path, target_path))

	template_image = ants.image_read(template_path)
	template_segmentations = ants.image_read(template_segmentation_path)

	if flip:
		template_image = flip_image(template_image)
		template_segmentations = flip_image(template_segmentations, single_components=True)

	target_image = ants.image_read(target_path)

	# if initial_transform is not None: 
	# 	target_mage = ants.apply_transforms(target_image, target_image, [initial_transform], verbose=True)

	if target_mask_path is not None:
		print("reading in mask", target_mask_path)
		target_mask = ants.image_read(target_mask_path)
		if target_mask.components > 1: 
			print('creating mask from segmentation')
			target_mask = create_mask(target_mask_path)
		else:
			print("creating mask from inverted region")
			target_mask = invert_mask(target_mask)
		print("---" * 20)

	if downsample:
		print("downsampling images")
		target_image = ants.resample_image(
			target_image, (downsample_size, downsample_size, downsample_size), 1, 0)
		template_image = ants.resample_image(
			template_image, (downsample_size, downsample_size, downsample_size), 1, 0)
		print("---" * 20)

	# transform_forward_affine = ants.registration(fixed=target_image, moving=template_image, mask=target_mask,
	# 											initial_transform=initial_transform,
	# 											type_of_transform="Affine", 
	# 											verbose=True
	# 											)

	transform_forward_syn = ants.registration(fixed=target_image, moving=template_image, mask=target_mask, 
											initial_transform=initial_transform,
											# initial_transform="identity",
											# initial_transform=transform_forward_affine['fwdtransforms'],
											type_of_transform="SyNOnly", 
											syn_metric="demons", 
											reg_iterations=(80, 40, 0), 
											verbose=True
											)

	print(transform_forward_syn)

	gc.collect()

	individual_segments = ants.split_channels(template_segmentations)
	predicted_targets = []

	forward_transforms = transform_forward_syn['fwdtransforms']

	if initial_transform is not None: 
		if initial_transform not in transform_forward_syn['fwdtransforms']: 
			forward_transforms = [transform_forward_syn['fwdtransforms'][0]] + [initial_transform]
		else: 
			forward_transforms = transform_forward_syn['fwdtransforms'][:2]

	# forward_transforms = transform_forward_affine['fwdtransforms'] + transform_forward_syn['fwdtransforms'][0]

	print(forward_transforms)
	print("---" * 20)

	for i, segment, in enumerate(individual_segments):

		print("applying transformation to template segmentations, segment %d"%i)
		predicted_target = ants.apply_transforms(target_image, segment, forward_transforms, interpolator="genericLabel", verbose=True)

		# predicted_target = ants.apply_transforms(target_image, segment, transform_forward_syn["fwdtransforms"], interpolator="genericLabel", verbose=True)

		predicted_targets.append(predicted_target)

	predicted_targets_image = ants.merge_channels(predicted_targets)

	print(predicted_targets_image.shape)
	print("writing out transformed template segmentations")


	if iteration is not None:
		if downsample:
			predicted_targets_path = os.path.join(save_dir, "%s%s-downsample%d-syn80-demons-%d.seg.nrrd" % (prefix, targ_id, downsample_size, iteration))
		else:
			predicted_targets_path = os.path.join(save_dir, "%s%s-syn80-demons-%d.seg.nrrd" % (prefix, targ_id, iteration))
	else:
		if downsample:
			predicted_targets_path = os.path.join(save_dir, "%s%s-downsample%d-syn80-demons.seg.nrrd" % (prefix, targ_id, downsample_size))
		else:
			predicted_targets_path = os.path.join(save_dir, "%s%s-syn80-demons.seg.nrrd" % (prefix, targ_id))

	target_image_header = nrrd.read_header(target_path)
	template_segmentations_header = nrrd.read_header(template_segmentation_path)

	# ants_image_to_file(predicted_targets_image, template_segmentations_header, target_image_header, predicted_targets_path)

	ants.image_write(predicted_targets_image, predicted_targets_path)

	return


def main():

	args = parse_command_line(sys.argv)

	base = args['base']
	template_path = args['template']
	template_segmentation_path = args['template_seg']
	target_path = args['target']
	target_mask_path = args['target_mask']

	dry_run = args['dry']
	downsample_image = args['downsample']
	flip = args['flip']

	save_dir = os.path.join(base, 'predictions')

	prefix = args['output_prefix']
	targ_id = args['target_scan_id']

	register_to_target(template_path, template_segmentation_path, target_path, base, save_dir, prefix, targ_id, 
		target_mask_path=target_mask_path, initial_transform=args['initial_transform'], flip=flip, dry=dry_run, downsample=downsample_image)

	return


if __name__ == "__main__":
	main()