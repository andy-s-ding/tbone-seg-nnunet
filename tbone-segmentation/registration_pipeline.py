"""
registration_pipeline.py

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

from utils.file_io import ants_image_to_file

def parse_command_line(args):
	'''

	'''

	print('parsing command line')

	parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
	parser.add_argument('--template', 
						action="store", 
						type=str, 
						default="144"
						)
	parser.add_argument('--side', 
						action="store", 
						type=str, 
						default="RT"
						)
	parser.add_argument('--base', 
						action="store", 
						type=str, 
						default="/scratch/groups/rtaylor2/ANTs-registration"
						)
	parser.add_argument('--target',
						action="store",
						type=str,
						default="146",
						help="specify which scan should be the target." 
					   )
	parser.add_argument('--dry',
                     action="store_true"
                     )
	parser.add_argument('--downsample',
						action="store_true"
					   )
	parser.add_argument('--mask',
						action="store",
						type=str,
						help='if a file is specified, it will be read in and used as a mask for the SyN registration')
	
	args = vars(parser.parse_args())
	return args


def register_to_target(template, target, base, side, save_dir, dry=False, downsample=False, iteration=None, downsample_size=300):
	'''
	
	'''

	print("---"*10)
	print("entering registration w/ template %s and target %s" % (template, target))

	template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")
	target_path = os.path.join(base, "images", side + ' ' + target + ".nrrd")
	template_segmentation_path = os.path.join(base, "segmentations/Segmentation %s %s-new.seg.nrrd" % (side, template))

	if dry:
		pass

	if not dry:
		target_image = ants.image_read(target_path)
		template_image = ants.image_read(template_path)
		template_segmentations = ants.image_read(template_segmentation_path)

		if downsample:
			target_image = ants.resample_image(
				target_image, (downsample_size, downsample_size, downsample_size), 1, 0)
			template_image = ants.resample_image(
				template_image, (downsample_size, downsample_size, downsample_size), 1, 0)

		transform_forward = ants.registration(fixed=target_image, moving=template_image,
		                                      type_of_transform="SyN", syn_metric="demons", reg_iterations=(80, 40, 0), verbose=True)

		gc.collect()

		individual_segments = ants.split_channels(template_segmentations)
		predicted_targets = []

		for i, segment, in enumerate(individual_segments):

			print("applying transformation to template segmentations, segment %d"%i)
			predicted_target = ants.apply_transforms(target_image, segment, transform_forward["fwdtransforms"], interpolator="genericLabel", verbose=True)

			predicted_targets.append(predicted_target)

		predicted_targets_image = ants.merge_channels(predicted_targets)

		print(predicted_targets_image.shape)
		print("writing out transformed template segmentations")
		if iteration is not None:
			if downsample:
				ants.image_write(predicted_targets_image, os.path.join(
					save_dir, "%s %s %s-downsample%d-syn80-demons-%d.seg.nrrd" % (side, template, target, downsample_size, iteration)))
			else:
				ants.image_write(predicted_targets_image, os.path.join(
					save_dir, "%s %s %s-syn80-demons-%d.seg.nrrd" % (side, template, target, iteration)))
		else:
			if downsample:
				ants.image_write(predicted_targets_image, os.path.join(
					save_dir, "%s %s %s-downsample%d-syn80-demons.seg.nrrd" % (side, template, target, downsample_size)))
			else:
				ants.image_write(predicted_targets_image, os.path.join(
					save_dir, "%s %s %s-syn80-demons.seg.nrrd" % (side, template, target)))
	return


def main():

	args = parse_command_line(sys.argv)

	side = args['side']
	base = args['base']
	template = args['template']
	dry_run = args['dry']
	target = args['target']
	downsample_image = args['downsample']

	images = os.path.join(base, 'images')
	save_dir = os.path.join(base, 'predictions')
	
	scan_id = ['142', 
			   '142-average',
			   '143', 
			   '144', 
			   '145', 
			   '146', 
			   '147', 
			   '152', 
			   '153'
			  ]
	
	if target is not None: 
		if target not in scan_id: 
			print('incorrectly specified target scan')
			return
		scan_id = [target]

	# for target in scan_id:
	# 	if template in target:
	# 		continue

	register_to_target(template, target, base, side, save_dir, dry=dry_run, downsample=downsample_image)

	return


if __name__ == "__main__":
	main()