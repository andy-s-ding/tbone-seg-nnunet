import os
import glob
import random
import pickle
import sys
import pickle as pkl
import argparse
import datetime
import json
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import *


# Establish labels
label_key = {
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


def rename_file(idx, path, file_type):
    # Rename file into nnUNet format
    if file_type == 'image':
        renamed_file = os.path.join(path, "jhu_" + '{0:0>3}'.format(idx) + "_0000.nii.gz")
    elif file_type == 'label':
        renamed_file = os.path.join(path, "jhu_" + '{0:0>3}'.format(idx) + ".nii.gz")
    return renamed_file


def return_label(file_id, args, dirpath=None):
    # Return label path given corresponding image file
    if args.dataset == 'original':
        label = os.path.join(dirpath, "reg_Segmentation_" + file_id+ ".nii.gz")
        
    if args.dataset == 'generated':
        if "deform" in file_id:
            dirpath = args.generated_label_dir
            label = os.path.join(dirpath, "Segmentation_" + os.path.basename(file_id))
        else:
            dirpath = args.original_dataset_dir
            label = os.path.join(dirpath, "reg_Segmentation_" + os.path.basename(file_id).split('reg_')[-1] )
    return label

def return_image(file_id, dirpath):
    # Return registered image path given corresponding subject ID
    image = os.path.join(dirpath, "reg_" + file_id + ".nii.gz")
    return image

def get_identifiers_from_splitted_files(folder: str):
    # nnUNet generate_dataset_json helper function
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = []
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))
    

class FilestructureArgParser(object):
    """Arg Parser for Filestructure File."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ArgParser for Filestructure Setup')
        self.parser.add_argument('--dataset', type=str, choices=('original','generated'),
                                 help='Which dataset to use.')
        self.parser.add_argument('--original_dataset_dir', type=str, 
                                 help='Directory with registered original files')
        self.parser.add_argument('--generated_dataset_dir', type=str, required=False,
                                 help='Directory with SSM generated files')
        self.parser.add_argument('--generated_label_dir', type=str, required=False,
                                 help='Directory with SSM generated labels')
        self.parser.add_argument('--pickle_path', type=str, help='Path to pickle file')
        self.parser.add_argument('--output_dir', type=str, help='Output directory')
        self.parser.add_argument('--task_num', type=str, help='Task #')

    def parse_args(self):
        args = self.parser.parse_args()

        # Save args to a JSON file
        json_save_dir = './datasplit_jsons'
        if not os.path.isdir(json_save_dir):
            os.mkdir(json_save_dir)
        date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(json_save_dir, '{}_{}'.format(args.dataset, date_string))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir = save_dir

        return args


def main(args):
    
    if args.dataset == 'original':
        data_path = args.original_dataset_dir
        csv_name = 'original_mapping.csv'
    else:
        csv_name = 'generated_mapping.csv'
    pkl_path = args.pickle_path
    target_dir = args.output_dir
    
    # Load in datasplit.pkl
    with open(pkl_path, "rb") as pickle_in:
        split = pkl.load(pickle_in)
    train_files = split['Train']
    test_files = split['Test']
    print(f"There are {len(train_files)} and {len(test_files)} test files.")
    
    
    # Establish filenames
    base_dir = os.path.join(target_dir, "nnUnet")
    task_dir = os.path.join(base_dir, "nnUNet_raw_data_base", "nnUNet_raw_data", f"Task{args.task_num}_TemporalBone")
    train_dir = os.path.join(task_dir, "imagesTr")
    train_label_dir = os.path.join(task_dir, "labelsTr")
    test_dir = os.path.join(task_dir, "imagesTs")
    test_label_dir = os.path.join(task_dir, "labelsTs")
    
    column_names = ['Original File', 'Mapping']
    mapping_df = pd.DataFrame(columns = column_names)
    
    # Check if files already exist, if not, then cp files into correct folders
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
        os.mkdir(os.path.join(base_dir, "nnUNet_raw_data_base"))
        os.mkdir(os.path.join(base_dir, "nnUNet_raw_data_base", "nnUNet_raw_data"))
        os.mkdir(task_dir)
        os.mkdir(train_dir)
        os.mkdir(train_label_dir)
        os.mkdir(test_dir)
        os.mkdir(test_label_dir)

        if args.dataset == 'original':
            print("Copying Train Files Over...")
            for i, file in tqdm(enumerate(train_files)):
                orig_img = return_image(file, data_path)
                new_img = rename_file(i, train_dir, 'image')
                mapping_df = mapping_df.append(pd.DataFrame({column_names[0]: [orig_img], column_names[1]: [new_img]}))

                command = f"cp {orig_img} {new_img}"
                # print(command)
                os.system(command)
                command = f"cp {return_label(file, args, dirpath=data_path)} {rename_file(i, train_label_dir, 'label')}"
                # print(command)
                os.system(command)
            print("Copying Test Files Over...")
            for j, file in tqdm(enumerate(test_files, i+1)):
                orig_img = return_image(file, data_path)
                new_img = rename_file(j, test_dir, 'image')
                mapping_df = mapping_df.append(pd.DataFrame({column_names[0]: [orig_img], column_names[1]: [new_img]}))

                command = f"cp {orig_img} {new_img}"
                # print(command)
                os.system(command)
                command = f"cp {return_label(file, args, dirpath=data_path)} {rename_file(j, test_label_dir, 'label')}"
                # print(command)
                os.system(command)

        elif args.dataset == 'generated':
            print("Copying Train Files Over...")
            for i, file in tqdm(enumerate(train_files)):
                new_img = rename_file(i, train_dir, 'image')
                mapping_df = mapping_df.append(pd.DataFrame({column_names[0]: [file], column_names[1]: [new_img]}))

                command = f"cp {file} {new_img}"
                # print(command)
                os.system(command)
                command = f"cp {return_label(file, args)} {rename_file(i, train_label_dir, 'label')}"
                # print(command)
                os.system(command)
            print("Copying Test Files Over...")
            for j, file in tqdm(enumerate(test_files, i+1)):
                new_img = rename_file(j, test_dir, 'image')
                mapping_df = mapping_df.append(pd.DataFrame({column_names[0]: [file], column_names[1]: [new_img]}))

                command = f"cp {file} {new_img}"
                # print(command)
                os.system(command)
                command = f"cp {return_label(file, args)} {rename_file(j, test_label_dir, 'label')}"
                # print(command)
                os.system(command)
                
        # Make dataset.json file
        mapping_df.to_csv(os.path.join('.', csv_name))
        generate_dataset_json(join(task_dir, 'dataset.json'), train_dir, test_dir, ("CT",),
                          labels=label_key, dataset_name=f"Task{args.task_num}_TemporalBone", license='hands off!')
    else:
        print(f"{base_dir} already exists.")


if __name__ == '__main__':
    parser = FilestructureArgParser()
    args_ = parser.parse_args()
    main(args_)
    ## For augmented dataset:
    # usage: python create_nnunet_filestructure.py --dataset generated --original_dataset_dir ../registered_niftis/ --generated_dataset_dir ../nii_files --generated_label_dir ../NIFTI_Segmentations --output_dir ../temp_jsoong/ --pickle_path ./datasplit_generated.pkl --task_num 999
    ## For original dataset:
    # usage: python create_nnunet_filestructure.py --dataset original --original_dataset_dir ../registered_niftis/ --output_dir ../temp_jsoong/ --pickle_path ./datasplit.pkl, --task_num 999
