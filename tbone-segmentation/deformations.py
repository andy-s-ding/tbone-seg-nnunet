'''
deformations.py

Entry point primarily for averaging deformation fields  
in order to build average template models. 

Has functionality for applying the average deformation field
to an image and its segmentation.

 '''

import numpy as np
import ants
import os 
import sys 
import argparse 
import nibabel as nib
import shutil
import psutil
import gc
import nrrd

from propagate_segments import apply_transform_to_image
from utils.file_io import ants_image_to_file


def parse_command_line(args):
	'''

	'''

	print('parsing command line')

	parser = argparse.ArgumentParser(description='Analysis of deformation fields')
	parser.add_argument('--template', 
						action="store", 
						type=str, 
						default="140"
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
	parser.add_argument('--dry',
						action="store_true"
					   )
	parser.add_argument('--average',
						action="store_true"
						)
	parser.add_argument('--cached', 
						action="store",
						type=str
						)
	parser.add_argument('--apply_image',
						action="store_true"
						)
	parser.add_argument('--apply_segments',
						action="store_true"
						)
	parser.add_argument('--apply_annotations',
						action="store_true"
						)
	
	args = vars(parser.parse_args())
	return args


def average(template, base, side, save_dir, dry=False):
	"""
	
	"""
	scan_id = ['138', 
			   '142',
			   '143', 
			   '144', 
			   '145', 
			   '146', 
			   '147', 
			   '152', 
			   '153'
			  ]

	print('-- entering average computation')
	
	initial_path = os.path.join(save_dir, '%s %s %s inverse.nii.gz'%(side, template, scan_id[0]))
	initial_nifti = nib.load(initial_path)
	process = psutil.Process(os.getpid())
	print(process.memory_info().rss, process.memory_info().vms)
	
	initial = initial_nifti.get_fdata()
	
	for scan in scan_id[1:]:
		if template in scan: 
			continue
			
		print(' including %s in the moving average'%scan)
		
		deform_path = os.path.join(save_dir, '%s %s %s inverse.nii.gz'%(side, template, scan))
		
		field_nifti = nib.load(deform_path)
		field = field_nifti.get_fdata()
		
		initial = np.sum(np.stack((initial, field), axis=-1), axis=-1)
		
		del field_nifti
		del field
		gc.collect()
		
		process = psutil.Process(os.getpid())
		print(process.memory_info().rss)
		
	
	out = nib.Nifti1Image(np.divide(initial, float(len(scan_id))), initial_nifti.affine, initial_nifti.header)
	out_path = os.path.join(save_dir, '%s_%s_average.nii.gz'%(side, template))
	nib.save(out, out_path)
	
	return out_path


def main():
	
	args = parse_command_line(sys.argv)
	
	side = args['side']
	base = args['base']
	template = args['template']
	dry_run = args['dry']
	
	images = os.path.join(base, 'images')
	# save_dir = os.path.join(base, 'transforms')
	save_dir = "/scratch/groups/rtaylor2/ANTs-registration/transforms/SyN_Demons_1.9/deform/"
	
	if args['average']: 
		transform_path = average(template, base, side, save_dir, dry=dry_run)
	else: 
		transform_path = args["cached"]

		if not os.path.exists(args["cached"]): 
			print("could not find cached file at %s" % args["cached"])
			return 

	if args['apply_image']: 

		# read in the image 
		template_image = ants.image_read(os.path.join(images, "%s %s.nrrd" % (side, template)))
		
		# transform it 
		transformed_template_image = apply_transform_to_image(transform_path, template_image, template_image, interpolator='linear')

		# write it out 
		ants.image_write(transformed_template_image, os.path.join(images, "%s %s average.nrrd"))

	if args['apply_segments']: 

		# read in segmentation + header
		template_segmentations = ants.image_read(
									os.path.join(base, "segmentations", "Segmentation %s %s.seg.nrrd" % (side, template)),
									pixeltype="unsigned char"
									)
		template_segmentations_header = nrrd.read_header(os.path.join(base, "segmentations", "Segmentation %s %s.seg.nrrd" % (side, template)))

		# read in target image + header (which is really just the original image and header)
		template_image = ants.image_read(os.path.join(images, "%s %s.nrrd" % (side, template)))
		template_image_header = nrrd.read_header(os.path.join(images, "%s %s.nrrd" % (side, template)))

		# transform the segmentations according to the average deformation field
		transformed_template_segs = apply_transform_to_image(transform_path, template_image, template_segmentations)

		# write out the transformed segmentations
		out_file = os.path.join(base, "segmentations", "Segmentation %s %s average.seg.nrrd" % (side, template))
		ants_image_to_file(transformed_template_segs, template_segmentations_header, template_image_header, out_file)

	if args['apply_annotations']: 
		template_image = ants.image_read(os.path.join(images, "%s %s.nrrd" % (side, template)))
		template_image_header = nrrd.read_header(os.path.join(images, "%s %s.nrrd" % (side, template)))

		template_annotations = ants.image_read(
									os.path.join(base, "annotations/Annotations %s %s.seg.nrrd" % (side, template))
									)
		template_annotations_header = nrrd.read_header(os.path.join(base, "annotations/Annotations %s %s.seg.nrrd" % (side, template)))

		transformed_annotations = apply_transform_to_image(transform_path, template_image, template_annotations)

		out_file = os.path.join(base, "annotations/Annotations %s %s average.seg.nrrd" % (side, template))
		ants_image_to_file(transformed_annotations, template_annotations_header, template_image_header, out_file)


						  
	return


if __name__ == "__main__":
	main()