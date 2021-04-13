import nibabel as nib
import numpy as np
import os
import glob
import sys
import torch
import nrrd

sys.path.append('../tbone-segmentation')
import utils.mesh_ops as mesh
import utils.metrics as metrics
import utils.file_io as io


def make_one_hot(img_np, num_classes):
    img_one_hot = np.zeros((num_classes, img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.int8)
    for i in range(num_classes):
        a = (img_np == i)
        print(a.shape)
        img_one_hot[i, :, :, :] = a

    return img_one_hot


def main(argv):
    path_truth = argv[0]
    path_pred = argv[1]

    if "nrrd" not in path_truth:
        print("Not .nrrd files, converting nifti to .nrrd...")
        header_truth = nib.load(path_truth)
        data_truth = header_truth.get_fdata()
        header_pred = nib.load(path_pred)
        data_pred = header_pred.get_fdata()
        print("Finding num classes...")

    else:
        print("Reading .nrrd files...")
        data_truth, header_truth = nrrd.read(path_truth)
        data_pred, header_pred = nrrd.read(path_pred)

    num_classes = np.unique(data_truth.ravel()).shape[0]

    if len(data_truth.shape) != 4:
        print("Converting GT to one hot...")
        # data_truth = io.convert_to_one_hot(data_truth, header_truth)
        # data_truth = torch.nn.functional.one_hot(torch.tensor(data_truth).long(), num_classes=num_classes).numpy()
        # data_truth = torch.tensor(data_truth).unsqueeze(0).long().cuda()
        data_truth = make_one_hot(data_truth, num_classes)
        # data_truth = np.transpose(data_truth, (3, 0, 1, 2))

    if len(data_pred.shape) != 4:
        print("Converting predictions to one hot...")
        # data_pred = io.convert_to_one_hot(data_pred, header_truth)
        # data_pred = torch.nn.functional.one_hot(torch.tensor(data_pred).long(), num_classes=num_classes).numpy()
        # data_pred = torch.tensor(data_pred).unsqueeze(0).long().cuda()
        data_pred = make_one_hot(data_pred, num_classes)
        # data_pred = np.transpose(data_pred, (3, 0, 1, 2))

    HD = metrics.calc_hausdorff(data_truth, header_truth, data_pred, header_pred,
                                indices=[i for i in range(num_classes - 1)], mesh_cache=None, prefix_truth=None,
                                prefix_pred=None)


if __name__ == '__main__':
    main(sys.argv[1:])

    # usage: python evaluate hausdorff.py gt_path pred_path
    # python evaluate_hausdorff.py ../fold_0_nrrd/GT/jhu_007.seg.nrrd ../fold_0_nrrd/preds/jhu_007.seg.nrrd
