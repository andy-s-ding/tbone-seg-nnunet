import os
import glob
import random
import pickle
import sys


def main(argv):
    random.seed(0)
    data_path = argv[0]
    
    excluded_ids = sorted(['153', '142', '152', '145', '151', '146', '169', '144', '150' '138'])
    SSM_templates = ['RT_153', 'RT_142', 'RT_152', 'LT_145', "LT_151, 'RT_146', 'RT_142', 'LT_169', 'LT_144', 'LT_151', 'RT_144', 'RT_150', 'RT_138']

    file_list = glob.glob(os.path.join(data_path, "reg_Segmentation_*.nii.gz"))

    # File IDs including left/right distinctions
    file_ids = [os.path.basename(file_name)[17:-7] for file_name in file_list]
    num_files = len(file_ids)
    # Non-duplicate number of ears (subjects)
    subject_ids = list(set([os.path.basename(file_name)[20:-7] for file_name in file_list]))
    subject_ids = [ID for ID in subject_ids if ID not in excluded_ids]
    num_subjects = len(set(subject_ids))

    print(subject_ids)
        
    # Dataset Split (cumulative sum):
    num_val = int(round(num_subjects*0.5))
    num_test = int(round(num_subjects*0.5))
    print("Number of val subjects: ", num_val, "\nNumber of testing subjects:", num_test, "\nTotal:", num_subjects)
    assert(num_val + num_test == num_subjects-1)

    random.shuffle(subject_ids)
    test_subjects = sorted(random.sample(subject_ids, num_test))
    val_subjects = sorted([subject for subject in subject_ids if subject not in test_subjects and subject not in excluded_ids])
    print("*"*10)
    print(excluded_ids)
    print(val_subjects)
    print(test_subjects)
    
    train_files = [f for f in file_ids if any(subject in f for subject in SSM_templates)]
    val_files = [f for f in file_ids if any(subject in f for subject in val_subjects)]
    test_files = [f for f in file_ids if any(subject in f for subject in test_subjects)]

    print("Number of train volumes: ", len(train_files), "\nNumber of val volumes: ", len(val_files), "\nNumber of testing subjects:", len(test_files))
    
    split = {'Train': train_files + val_files, 'Test': test_files}
    
    print(split)
    
    with open("datasplit.pkl", "wb") as pickle_out:
        pickle.dump(split, pickle_out, protocol=4) # compatible with python 3.6+
        
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python create_datasplit.py-ablation-3 ../registered_niftis/
