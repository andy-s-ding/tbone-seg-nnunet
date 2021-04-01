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

def adjust_file_path(save_dir, prefix, suffix, downsample=None, downsample_size=None, iteration=None, registration=None, is_annotation=False):

	path = os.path.join(save_dir, prefix)

	if downsample:
		path += "-downsample%d" % (downsample_size)

	if registration is not None:
		path += "-" + registration

	if iteration is not None:
		path += "-%d" % (iteration)

	if is_annotation:
		path += "-annotations"

	path += suffix

	print(" -- returning path: %s" % path)

	return path

base = '/Volumes/Extreme SSD/ANTs-registration'
template = '153'
side = 'RT'

save_dir = os.path.join(base, 'predictions')
images = os.path.join(base, 'images')

template_path = os.path.join(base, "images", side + ' ' + template + ".nrrd")

print("reading template image")
template_image_header = nrrd.read_header(template_path)
template_image = ants.image_read(template_path)

print("flipping template")
flipped_image = ants.reflect_image(template_image, axis=0)

print("writing out flipped template")

# adjust file path according to input parameters
flipped_image_path = adjust_file_path(save_dir, "%s %s-flipped"%(side, template), ".nrrd")

ants.image_write(flipped_image, flipped_image_path)

# ants_image_to_file(flipped_image, template_image_header, template_image_header, flipped_image_path)
