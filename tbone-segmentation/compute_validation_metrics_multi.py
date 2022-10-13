"""
compute_validation_metrics_multi.py

Compute Hausdorff distances and/or Dice Scores for segment ids

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
    5: "Bony_Labyrinth",
    6: "Vestibular_Nerve",
    7: "Superior_Vestibular_Nerve",
    8: "Inferior_Vestibular_Nerve",
    9: "Cochlear_Nerve",
    10: "Facial_Nerve",
    11: "Chorda_Tympani",
    12: "ICA",
    13: "Sinus_and_Dura",
    14: "Vestibular_Aqueduct",
    15: "Mandible",
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
    parser.add_argument('--model',
                        action="store",
                        type=str,
                        default='nnUNetTrainerV2'
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
    parser.add_argument('--hausdorff',
                        action="store_true",
                        help="True if evaluating modified Hausdorff distances"
                        )
    parser.add_argument('--hausdorff_out',
                        action="store",
                        type=str,
                        default="Hausdorff Distances",
                        help="Name of the .csv file containing Hausdorff distances that will be saved in --base"
                        )
    parser.add_argument('--dice',
                        action="store_true",
                        help="True if evaluating Dice scores"
                        )
    parser.add_argument('--dice_out',
                        action="store",
                        type=str,
                        default="Dice Scores",
                        help="Name of the .csv file containing Dice scores that will be saved in --base"
                        )
    parser.add_argument('--overwrite',
                        action="store_true",
                        help="True if overwriting output files"
                        )
    parser.add_argument('--raw',
                        action="store_true",
                        help="True if analyzing raw validation files"
                        )
    parser.add_argument('--pp',
                        action="store",
                        default="_postprocessed",
                        help="Suffix for folder containing postprocessed validation files"
                        )

    args = vars(parser.parse_args())
    print(args)
    return args


def main():
    args = parse_command_line(sys.argv)

    base = os.path.join(args['base'], f"nnUnet/nnUNet_trained_models/nnUNet/3d_fullres/Task{args['task_num']}_TemporalBone/{args['model']}__nnUNetPlansv2.1")
    gt_folder = os.path.join(base, 'gt_niftis')
    hausdorff = args['hausdorff']
    dice = args['dice']
    assert(hausdorff or dice), "Metric not specified!"

    # Determine output file paths and their existence
    hausdorff_path = os.path.join(base, f"{args['hausdorff_out']}.csv")
    dice_path = os.path.join(base, f"{args['dice_out']}.csv")

    if args['overwrite']: print("CAUTION: Will overwrite existing output files!")
    else:
        if hausdorff and os.path.exists(hausdorff_path):
            hausdorff = False
            print("Hausdorff output file exists. Will not perform Hausdorff analysis.")
        if dice and os.path.exists(dice_path):
            dice = False
            print("Dice output file exists. Will not perform Hausdorff analysis.")
    if not hausdorff and not dice:
        print("Existing output files present. Rerun with --overwrite to continue.")
        return

    # Determine segment ids to analyze
    if args['ids'] is None: # Do all segment ids
        ids = list(range(1,len(seg_names))) # Exclude background (segments are 1-indexed)
    else: ids = args['ids'] # Otherwise, do specified segment ids
    print(f"Segment IDs: {ids}")
    seg_name = '-'.join(seg_names[i] for i in ids)
    # Initialize metric dictionaries
    if hausdorff:
        hausdorff_dict = dict()
        hausdorff_dict['Fold'] = []
        hausdorff_dict['Target'] = []
        hausdorff_dict[seg_name] = []
    if dice:
        dice_dict = dict()
        dice_dict['Fold'] = []
        dice_dict['Target'] = []
        dice_dict[seg_name] = []

    # Determine fold numbers to analyze
    if args['folds'] is None: # Do all existing folds
        folds = [folder[-1] for folder in os.listdir(base) if fnmatch.fnmatch(folder, 'fold_*')]
    else: folds = args['folds'] # Otherwise, do specified folds

    # Determine which validation folder to analyze
    if args['raw']: val_dir = 'validation_raw'
    else: 
        pp_suffix = args['pp']
        val_dir = 'validation_raw' + pp_suffix

    for fold in folds:
        print(f"Fold: {fold}")
        val_path = os.path.join(base, f"fold_{fold}", val_dir)

    # try:
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

            combined_pred = np.logical_or.reduce(pred_one_hot.take(ids, axis=0))
            combined_gt = np.logical_or.reduce(gt_one_hot.take(ids, axis=0))

            # Update output metric dictionaries
            if hausdorff:
                hausdorff_dict['Fold'].append(fold)
                hausdorff_dict['Target'].append(target)
            if dice:
                dice_dict['Fold'].append(fold)
                dice_dict['Target'].append(target)

            print(f"---- Computing metrics for segment: {seg_name}")
            if hausdorff:
                surface_distances = compute_surface_distances(combined_gt, combined_pred, spacing)
                mod_hausdorff_distance = max(compute_average_surface_distance(surface_distances))
                print(f"------ Distance: {mod_hausdorff_distance}")
                hausdorff_dict[seg_name].append(mod_hausdorff_distance)
            if dice:
                dice_coeff = compute_dice_coefficient(combined_gt, combined_pred)
                print(f"------ Dice Score: {dice_coeff}")
                dice_dict[seg_name].append(dice_coeff)
    # except: break

    if hausdorff:
        hausdorff_df = pd.DataFrame.from_dict(hausdorff_dict)
        print('Modified Hausdorff Distances')
        print(hausdorff_df)
        hausdorff_df.to_csv(hausdorff_path)
    if dice:
        dice_df = pd.DataFrame.from_dict(dice_dict)
        print('Dice Scores')
        print(dice_df)
        dice_df.to_csv(dice_path)


    return


if __name__ == "__main__":
    main()

# internal use: python compute_validation_metrics.py --base '/media/andyding/SAMSUNG 4TB/tbone-seg-nnunet/02_ading' --model nnUNetTrainerV2 --dice --hausdorff
