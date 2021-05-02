"""
propagate_segments.py

Andy's copy of registration_pipeline with functionality for segment and annotation propagation 

"""
import os 
import sys 
import argparse 
import numpy as np
import ants
import nrrd

import shutil
import gc

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
	parser.add_argument('--registration',
						action="store_true",
						help="required if there are no cached deformations"
						)
	parser.add_argument('--cached',
						action="store",
						type=str, 
                     default="/Volumes/Extreme SSD/ANTs-registration/transforms"
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
	parser.add_argument('--transforms',
                     	action="store_true",
						help="determines whether to write the transforms of the deformation field"
                     	)
	parser.add_argument('--annotations',
                     	action="store_true",
						help="determines whether to propagate and write template annotations"
                     	)
	parser.add_argument('--no_segs',
                     	action="store_true",
						help="determines whether to only propagate annotations"
                     	)
	parser.add_argument('--overwrite',
                     	action="store_true",
						help="determines whether to overwrite existing output .seg.nrrds if present"
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


def propagate(template, target, base, side, save_dir, transform, downsample=False, downsample_size=300, propagate_segmentations=False, propagate_annotations=False, flip=False, nifti=False):  
	if side == 'RT': other_side = 'LT'
	else: other_side = 'RT'

	if flip: target_path = os.path.join(base, "images", other_side + ' ' + target + ".nrrd")
	else: target_path = os.path.join(base, "images", side + ' ' + target + ".nrrd")

	target_image = ants.image_read(target_path)
	target_image_header = nrrd.read_header(target_path)

	propagated = []

	if propagate_segmentations: 
		template_segmentation_path = os.path.join(base, "segmentations/Segmentation %s %s.seg.nrrd" % (side, template))
		template_segmentations = ants.image_read(template_segmentation_path, pixeltype="unsigned char")
		template_segmentations_header = nrrd.read_header(template_segmentation_path)

		if flip:
			template_segmentations = flip_image(template_segmentations, single_components=True)

		predicted_segmentations = apply_transform_to_image(transform, target_image, template_segmentations)
		predicted_targets_path = adjust_file_path(save_dir, "Segmentation %s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample, downsample_size=downsample_size, flip=flip)
		ants_image_to_file(predicted_segmentations, template_segmentations_header, target_image_header, predicted_targets_path)
		
		if nifti:
			predicted_nii_path = adjust_file_path(os.path.join(save_dir, 'NIFTI Predictions'), "Segmentation %s %s %s"%(side, template, target), ".nii.gz", downsample, downsample_size, flip=flip)
			ants_image_to_file(predicted_segmentations, template_segmentations_header, target_image_header, predicted_nii_path, nifti=True)

		propagated.append(predicted_segmentations)

	if propagate_annotations: 
		template_annotation_path = os.path.join(base, "annotations/Annotations %s %s.seg.nrrd" % (side, template))
		template_annotations = ants.image_read(template_annotation_path, pixeltype="unsigned char")
		template_annotations_header = nrrd.read_header(template_annotation_path)

		if flip:
			template_annotations = flip_image(template_annotations)

		predicted_annotations = apply_transform_to_image(transform, target_image, template_annotations)
		predicted_annotations_path = adjust_file_path(save_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", is_annotation=True, downsample=downsample, downsample_size=downsample_size, flip=flip)
		ants_image_to_file(predicted_annotations, template_annotations_header, target_image_header, predicted_annotations_path)

		propagated.append(predicted_annotations)

	return propagated


def register_to_target(template, target, base, side, save_dir, downsample=False, downsample_size=300, write_transforms=False, write_annotations=False, flip=False, nifti=False): 
	"""Summary
	
	Args:
	    template (TYPE): Description
	    target (TYPE): Description
	    base (TYPE): Description
	    side (TYPE): Description
	    save_dir (TYPE): Description
	    dry (bool, optional): Description
	    downsample (bool, optional): Description
	    downsample_size (int, optional): Description
	    write_transforms (bool, optional): Description
	    write_annotations (bool, optional): Description
	
	Returns:
	    TYPE: Description
	"""
	print("---"*10)
	print("entering registration w/ template %s and target %s"%(template, target))
	
	if side == 'RT': other_side = 'LT'
	else: other_side = 'RT'

	template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")
	if flip: target_path = os.path.join(base, "images", other_side + ' ' + target + ".nrrd")
	else: target_path = os.path.join(base, "images", side + ' ' + target + ".nrrd")

	# Get header information
	template_segmentation_path = os.path.join(base, "segmentations/Segmentation %s %s.seg.nrrd" % (side, template))
	template_segmentations_header = nrrd.read_header(template_segmentation_path)
	target_image_header = nrrd.read_header(target_path)

	# Make ANTsImages
	target_image = ants.image_read(target_path)
	template_image = ants.image_read(template_path)
	template_segmentations = ants.image_read(template_segmentation_path, pixeltype="unsigned char")

	if flip: # Flip template image if target image is on contralateral side
		template_image = flip_image(template_image)
		template_segmentations = flip_image(template_segmentations, single_components=True)

	if downsample: # Downsample to speed up registration
		target_image_downsample = ants.resample_image(target_image, (downsample_size, downsample_size, downsample_size), 1, 0)
		template_image_downsample = ants.resample_image(template_image, (downsample_size, downsample_size, downsample_size), 1, 0)
		transform_forward = ants.registration(fixed=target_image_downsample, moving=template_image_downsample, type_of_transform="SyN", syn_metric="demons", reg_iterations=(80,40,0), verbose=True)
	
	else:
		transform_forward = ants.registration(fixed=target_image, moving=template_image, type_of_transform="SyN", syn_metric="demons", reg_iterations=(80,40,0), verbose=True)

	print("propagating segments")
	predicted_targets_image = apply_transform_to_image(transform_forward["fwdtransforms"], target_image, template_segmentations)

	print(predicted_targets_image.shape)
	print("writing out transformed template segmentations")

	# adjust file path according to input parameters
	predicted_targets_path = adjust_file_path(save_dir, "Segmentation %s %s %s"%(side, template, target), ".seg.nrrd", downsample, downsample_size, flip=flip)
	ants_image_to_file(predicted_targets_image, template_segmentations_header, target_image_header, predicted_targets_path)

	if nifti:
		predicted_nii_path = adjust_file_path(os.path.join(save_dir, 'NIFTI Predictions'), "Segmentation %s %s %s"%(side, template, target), ".nii.gz", downsample, downsample_size, flip=flip)
		ants_image_to_file(predicted_targets_image, template_segmentations_header, target_image_header, predicted_nii_path, nifti=True)

	if write_annotations:
		print("writing annotations")
		template_annotation_path = os.path.join(base, "annotations/Annotations %s %s.seg.nrrd" % (side, template))
		template_annotations = ants.image_read(template_annotation_path)
		template_annotations_header = nrrd.read_header(template_annotation_path)

		if flip: template_annotations = flip_image(template_annotations)

		predicted_annotations_image = apply_transform_to_image(transform_forward["fwdtransforms"], target_image, template_annotations)

		print(predicted_annotations_image.shape)

		predicted_annotations_path = adjust_file_path(save_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", downsample, downsample_size, is_annotation=True, flip=flip)
		ants_image_to_file(predicted_annotations_image, template_annotations_header, target_image_header, predicted_annotations_path)

	if write_transforms:
		print("writing transforms")
		transform_dir = os.path.join(base, "transforms")

		affine_path = adjust_file_path(transform_dir, "%s %s %s" % (side, template, target), ".mat", registration='affine',
							downsample=downsample, downsample_size=downsample_size, flip=flip)
		deform_path = adjust_file_path(transform_dir, "%s %s %s" % (side, template, target), ".nii.gz", registration='forward',
							downsample=downsample, downsample_size=downsample_size, flip=flip)

		shutil.move(transform_forward['fwdtransforms'][0], deform_path)
		shutil.move(transform_forward['fwdtransforms'][1], affine_path)

	gc.collect()
		
	return 


def main(): 
	args = parse_command_line(sys.argv)

	side = args['side']
	base = args['base']
	template = args['template']
	target = args['target']
	images = os.path.join(base, 'images')
	save_dir = os.path.join(base, 'predictions')
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
			predicted_targets_path = adjust_file_path(save_dir, "Segmentation %s %s %s"%(side, template, target), ".seg.nrrd", args['downsample'], args['downsample_size'], flip=args['flip'])
			predicted_annotations_path = adjust_file_path(save_dir, "Segmentation %s %s %s"%(side, template, target), ".seg.nrrd", args['downsample'], args['downsample_size'], is_annotation=True, flip=args['flip'])
			if not args['overwrite'] and os.path.exists(predicted_targets_path) and os.path.exists(predicted_annotations_path):
				print('output .seg.nrrds already exist')
				continue

			if args['registration']:
				register_to_target(template, target, base, side, save_dir, 
						dry=args['dry'], downsample=args['downsample'], downsample_size=args['downsample_size'],
						write_transforms=args['transforms'], write_annotations=args['annotations'], flip=args['flip'], nifti=args['nifti'])

			else: 
				affine_path = adjust_file_path(args["cached"], "%s %s %s" % (side, template, target), ".mat", registration='affine',
                                   downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])
				deform_path = adjust_file_path(args["cached"], "%s %s %s" % (side, template, target), ".nii.gz", registration='forward',
                                   downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'])

				if not (os.path.exists(deform_path) and os.path.exists(affine_path)): 
					print('-- expected cached results at %s, %s' % (affine_path, deform_path))
					return

				need_segmentations = not args['no_segs']

				if not args['overwrite']:
					need_segmentations = need_segmentations and not os.path.exists(predicted_targets_path)
					need_annotations = args['annotations'] and not os.path.exists(predicted_annotations_path)
					propagate(template, target, base, side, save_dir, [deform_path, affine_path], 
                            propagate_segmentations=need_segmentations, propagate_annotations=need_annotations,
							downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'], nifti=args['nifti'])
				
				else: propagate(template, target, base, side, save_dir, [deform_path, affine_path], 
							propagate_segmentations=need_segmentations, propagate_annotations=True,
                            downsample=args['downsample'], downsample_size=args['downsample_size'], flip=args['flip'], nifti=args['nifti'])

	return 


if __name__ == "__main__":
	main()
