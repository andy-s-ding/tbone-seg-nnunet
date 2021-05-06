"""
compute_accuracy_metrics.py

Compute dice scores and Hausdorff distances for segment ids

"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
import surface_distance
from surface_distance.metrics import *
from utils.file_io import *

seg_names = {
    0: "Background",
    1: "Bone",
    2: "Malleus",
    3: "Incus",
    4: "Stapes",
    5: "Vestibule_and_Cochlea",
    6: "Vestibular_Nerve",
    7: "Superior_Vestibular_Nerve",
    8: "Inferior_Vestibular_Nerve",
    9: "Cochlear_Nerve",
    10: "Facial_Nerve",
    11: "Chorda_Tympani",
    12: "ICA",
    13: "Sinus_and_Dura",
    14: "Vestibular_Aqueduct",
    15: "TMJ",
    16: "EAC",
}

def parse_command_line(args):
    '''

	'''

    print('parsing command line')

    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
    parser.add_argument('--base',
                        action="store",
                        type=str,
                        default="/Volumes/Extreme SSD/ANTs-registration"
                        )
    parser.add_argument('--prediction',
                        action="store",
                        type=str,
                        )
    parser.add_argument('--target',
                        action="store",
                        type=str,
                        help="specify which scan should be the target scan. \
                        If a scan is specified, only the registration of the specified target scan onto the template scan will be performed. \
                        If the target scan is not specified, the entire list of scans will be registered onto the template."
                        )
    parser.add_argument('--target_nrrd',
                        action="store",
                        type=str,
                        )
    parser.add_argument('--fold',
                        action="store",
                        type=str,
                        )
    parser.add_argument('--dry',
                        action="store_true"
                        )

    args = vars(parser.parse_args())
    return args


def main():
    args = parse_command_line(sys.argv)

    base = args['base']
    prediction = args['prediction']
    target = args['target']
    ids = [i for i in range(0, 17)]
    print(ids)
    gt_nii_path = os.path.join(base, 'gt_niftis', target)
    pred_path = os.path.join(base, f"fold_{args['fold']}", 'validation_raw_postprocessed', prediction)
    gt_seg_path_nrrd = os.path.join(os.path.dirname(gt_nii_path), args['target_nrrd'])

    # Initialize metric dictionaries
    hausdorff_dict = dict()
    hausdorff_dict['Target'] = []
    for i in ids:  # seg_names are 0-indexed, while ids are 1-indexed (due to presence of background class in one-hot)
        hausdorff_dict[seg_names[i]] = []


    print('-- Evaluating %s' % (target))

    pred_seg = np.array(nib.load(pred_path).dataobj) #s
    gt_seg = np.array(nib.load(gt_nii_path).dataobj)
    spacing = np.linalg.norm(nrrd.read_header(gt_seg_path_nrrd)['space directions'][1:], axis=0)

    pred_one_hot = np.moveaxis((np.arange(pred_seg.max() + 1) == pred_seg[..., None]), -1, 0)
    gt_one_hot = np.moveaxis((np.arange(gt_seg.max() + 1) == gt_seg[..., None]), -1, 0)

    print(pred_one_hot.shape)
    print(gt_one_hot.shape)

    hausdorff_dict['Target'].append(target)

    for i in ids:
        seg_name = seg_names[i]
        print('---- Computing metrics for segment: %s' % seg_name)
        # breakpoint()
        surface_distances = compute_surface_distances(gt_one_hot[i], pred_one_hot[i], spacing)
        mod_hausdorff_distance = max(compute_average_surface_distance(surface_distances))

        hausdorff_dict[seg_name].append(mod_hausdorff_distance)

    hausdorff_df = pd.DataFrame.from_dict(hausdorff_dict)

    print('Modified Hausdorff Distances')
    print(hausdorff_df)

    hausdorff_path = os.path.join(base, f'{target[:8]}_hausdorff.csv')

    hausdorff_df.to_csv(hausdorff_path)

    return


if __name__ == "__main__":
    main()
