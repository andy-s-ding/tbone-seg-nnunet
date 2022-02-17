"""
compute_hausdorff_distance.py

Compute Hausdorff distances for segment ids

"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import fnmatch
import surface_distance
from surface_distance.metrics import *
from utils.file_io import *
import time

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
                        default="/media/andyding/SAMSUNG 4TB/tbone-seg-nnunet/01_ading"
                        )
    parser.add_argument('--task_num',
                        action="store",
                        type=int,
                        default=101
                        )
    parser.add_argument('--targets',
                        action="store",
                        type=str,
                        nargs='*',
                        help="Specify which scan should be the target scan. \
                        If a scan is specified, only the registration of the specified target scan onto the template scan will be performed. \
                        If the target scan is not specified, the entire list of scans will be registered onto the template."
                        )
    parser.add_argument('--folds',
                        action="store",
                        type=int,
                        nargs='*'
                        )
    parser.add_argument('--ids',
                        type=int,
                        nargs='+',
                        help="Segment indices (1-indexed) to calculate accuracy metrics. If no ids are specified, all foreground ids \
                        will be included.")
    parser.add_argument('--save_name',
                        action="store",
                        type=str,
                        default="Hausdorff Distances",
                        help="Name of the .csv file containing Hausdorff distances that will be saved in --base"
                        )

    args = vars(parser.parse_args())
    return args


def main():
    args = parse_command_line(sys.argv)

    base = os.path.join(args['base'], f"nnUnet/nnUNet_trained_models/nnUNet/3d_fullres/Task{args['task_num']}_TemporalBone/nnUNetTrainerV2__nnUNetPlansv2.1")
    gt_folder = os.path.join(base, 'gt_niftis')

    # Determine segment ids to analyze
    if args['ids'] is None: # Do all segment ids
        ids = list(range(1,len(seg_names))) # Exclude background
    else: ids = args['ids'] # Otherwise, do specified segment ids
    print(f"Segment IDs: {ids}")
    
    # Initialize metric dictionaries
    hausdorff_dict = dict()
    hausdorff_dict['Fold'] = []
    hausdorff_dict['Target'] = []
    for i in ids:  # seg_names are 0-indexed, while ids are 1-indexed (due to presence of background class in one-hot)
        hausdorff_dict[seg_names[i]] = []

    # Determine fold numbers to analyze
    if args['folds'] is None: # Do all existing folds
        folds = [folder[-1] for folder in os.listdir(base) if fnmatch.fnmatch(folder, 'fold_*')]
    else: folds = args['folds'] # Otherwise, do specified folds

    for fold in folds:
        print(f"Fold: {fold}")
        val_path = os.path.join(base, f"fold_{fold}", 'validation_raw_postprocessed')

        # Determine targets in fold
        if args['targets'] is None: # Do all targets
            targets = [target.split('.nii.gz')[0] for target in os.listdir(val_path) if fnmatch.fnmatch(target, '*.nii.gz')]
        else: targets = args['targets'] # Otherwise, do specified targets
        print(f"Targets in fold: {targets}")

        for target in targets:
            pred_path = os.path.join(val_path, f"{target}.nii.gz")
            gt_path = os.path.join(gt_folder, f"{target}.nii.gz")

            print('-- Evaluating %s' % (target))
            gt_nii = nib.load(gt_path)
            gt_seg = np.array(gt_nii.dataobj)
            gt_header = gt_nii.header
            pred_nii = nib.load(pred_path)
            pred_seg = np.array(pred_nii.dataobj)
            spacing = np.asarray(gt_header.get_zooms())
            print(f"---- Spacing for {target}: {spacing}")

            pred_one_hot = np.moveaxis((np.arange(pred_seg.max() + 1) == pred_seg[..., None]), -1, 0)
            gt_one_hot = np.moveaxis((np.arange(gt_seg.max() + 1) == gt_seg[..., None]), -1, 0)

            # Update output metric dictionary
            hausdorff_dict['Fold'].append(fold)
            hausdorff_dict['Target'].append(target)
            for i in ids:
                seg_name = seg_names[i]
                print(f"---- Computing metrics for segment: {seg_name}")
                surface_distances = compute_surface_distances(gt_one_hot[i], pred_one_hot[i], spacing)
                mod_hausdorff_distance = max(compute_average_surface_distance(surface_distances))
                print(f"------ Distance: {mod_hausdorff_distance}")
                hausdorff_dict[seg_name].append(mod_hausdorff_distance)

    hausdorff_df = pd.DataFrame.from_dict(hausdorff_dict)

    print('Modified Hausdorff Distances')
    print(hausdorff_df)

    hausdorff_path = os.path.join(base, f"{args['save_name']}.csv")
    hausdorff_df.to_csv(hausdorff_path)

    return


if __name__ == "__main__":
    main()

# internal use: python compute_hausdorff_distance.py