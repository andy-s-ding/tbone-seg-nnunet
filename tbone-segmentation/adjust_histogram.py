"""
adjust_histogram.py
Histogram equalization for iamges
"""
import sys 
import argparse 
import numpy as np

from skimage import exposure
import nibabel as nib

def parse_command_line(args):
    '''

    '''

    print('parsing command line')

    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
    parser.add_argument('--in',
                        action="store",
                        type=str
                        )
    parser.add_argument('--out',
                        action="store",
                        type=str,
                        default=None
                        )
    parser.add_argument('--method',
                        action="store",
                        type=str,
                        choices=['cs', 'histeq', 'adeq'],
                        default='cs'
                        )

    args = vars(parser.parse_args())
    print(args)
    return args


def adjust_histogram(input_path, method, output_path=None, ):
    input_image = nib.load(input_path)
    input_array = input_image.get_fdata()

    # Contrast stretching
    if method == "cs":
        p2, p98 = np.percentile(input_array, (2, 98))
        output_array = exposure.rescale_intensity(input_array, in_range=(p2, p98))

    # Equalization
    elif method == "histeq":
        output_array = exposure.equalize_hist(input_array)

    # Adaptive Equalization
    else:
        output_array = exposure.equalize_adapthist(input_array/np.max(abs(input_array)), clip_limit=0.03)

    output_image = nib.nifti1.Nifti1Image(output_array, input_image.affine, input_image.header)

    if output_path: nib.save(output_image, output_path)
    else: nib.save(output_image, input_path.split('.nii.gz')[0] + f"_{method}" + ".nii.gz")

    return

def main():
    args = parse_command_line(sys.argv)
    input_path = args['in']
    output_path = args['out']
    method = args['method']
    
    adjust_histogram(input_path, method, output_path)

if __name__ == '__main__':
    main()