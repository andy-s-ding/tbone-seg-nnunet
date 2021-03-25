import os
import glob
import random
import pickle
import sys


def main(argv):
    random.seed(0)
    data_path = argv[0]

    file_list = glob.glob(os.path.join(data_path, "reg_Segmentation_*.nii.gz"))

    # File IDs including left/right distinctions
    file_ids = [os.path.basename(file_name)[17:-7] for file_name in file_list]
    num_files = len(file_ids)
    # Non-duplicate number of ears (subjects)
    subject_ids = list(set([os.path.basename(file_name)[20:-7] for file_name in file_list]))
    num_subjects = len(set(subject_ids))

    # Dataset Split (cumulative sum):
    num_train = int(round(num_subjects*0.7))
    num_test = int(round(num_subjects*0.3))
    print("Number of training subjects: ", num_train, "\nNumber of testing subjects:", num_test, "\nTotal:", num_subjects)
    assert(num_train + num_test == num_subjects)

    random.shuffle(subject_ids)
    train_subjects = random.sample(subject_ids, num_train)
    test_subjects = [subject for subject in subject_ids if subject not in train_subjects]

    train_files = [f for f in file_ids if any(subject in f for subject in train_subjects)]
    test_files = [f for f in file_ids if any(subject in f for subject in test_subjects)]

    split = {'Train': train_files, 'Test': test_files}
    
    print(split)
    
    with open("datasplit.pkl", "wb") as pickle_out:
        pickle.dump(split, pickle_out, protocol=4) # compatible with python 3.6+
        
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python create_datasplit.py ../registered_niftis/