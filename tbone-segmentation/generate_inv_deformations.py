"""
generate_inv_deformations.py

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
						default="/media/andyding/EXTREME SSD/ANTs-registration/"
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
						default="/media/andyding/EXTREME SSD/ANTs-registration/transforms/"
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

def register_to_template(template, target, base, side, save_dir, downsample=False, downsample_size=300, flip=False):
	print("---"*10)
	print("entering registration w/ template %s and target %s" % (template, target))

	if side == 'RT':
		other_side = 'LT'
	else:
		other_side = 'RT'

	template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")
	if flip:
		target_path = os.path.join(base, "images", other_side + ' ' + target + ".nrrd")
	else:
		target_path = os.path.join(base, "images", side + ' ' + target + ".nrrd")

	affine_path = adjust_file_path(save_dir, "%s %s %s" % (side, template, target), ".mat", registration='affine-inverse',
                                downsample=downsample, downsample_size=downsample_size, flip=flip)
	deform_path = adjust_file_path(save_dir, "%s %s %s" % (side, template, target), ".nii.gz", registration='inverse',
                                downsample=downsample, downsample_size=downsample_size, flip=flip)

	target_image = ants.image_read(target_path)
	template_image = ants.image_read(template_path)

	if flip:
		target_image = flip_image(target_image)

	if downsample:
		target_image_downsample = ants.resample_image(target_image, (downsample_size, downsample_size, downsample_size), 1, 0)
		template_image_downsample = ants.resample_image(template_image, (downsample_size, downsample_size, downsample_size), 1, 0)
		transform_forward = ants.registration(fixed=template_image_downsample, moving=target_image_downsample,
												type_of_transform="SyN", syn_metric="demons", reg_iterations=(80, 40, 0), verbose=True)

	else:
		transform_forward = ants.registration(fixed=template_image, moving=target_image,
												type_of_transform="SyN", syn_metric="demons", reg_iterations=(80, 40, 0), verbose=True)

	print("writing transforms")
	shutil.move(transform_forward['invtransforms'][1], deform_path)
	shutil.move(transform_forward['invtransforms'][0], affine_path)

	gc.collect()

	return

def main(): 
	args = parse_command_line(sys.argv)
	side = args['side']
	base = args['base']
	template = args['template']
	target = args['target']
	save_dir = os.path.join(base, 'transforms')
	RT = [	'138', 
			'142',
			'143', 
			'144',
			'146', 
			'147',
			'152', 
			'153',
			'168',
			'170',
			'172',
			'174', 
			'175',
			'177',
			'179',
			'181',
			'183',
			'184',
			'186',
			'187',
			'189',
			'191',
			'192',
			'194',
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
			'168',
			'169', 
			'170', 
			'171', 
			'172',
			'173',
			'175',
			'176',
			'177',
			'181',
			'183',
			'184',
			'185',
			'186',
			'191',
			'192',
			'193',
			'194',
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

	for target in target_scan_id:
		if template in target: 
			continue
		else:
			affine_path = adjust_file_path(args["cached"], "%s %s %s" % (side, template, target), ".mat", registration='affine-inverse',
                                  			downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
			deform_path = adjust_file_path(args["cached"], "%s %s %s" % (side, template, target), ".nii.gz", registration='inverse',
                                  			downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])

			if args['overwrite'] or not(os.path.exists(deform_path) and os.path.exists(affine_path)):
				register_to_template(template, target, base, side, save_dir, downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
			
			else:
				print('-- transform already exists at %s, %s' % (affine_path, deform_path))
	return 


if __name__ == "__main__":
	main()
