import os
import glob
import random
import pickle
import sys


def main(argv):
    random.seed(0)
    original_datasplit = argv[0]
    original_data_path = argv[1]
    generated_data_path = argv[2]

    with open(original_datasplit, 'rb') as pickle_in:
        og_datasplit = pickle.load(pickle_in)

    original_train = og_datasplit['Train']
    original_train = [os.path.join(original_data_path, f"reg_{f}.nii.gz") for f in original_train]
    generated_train = glob.glob(os.path.join(generated_data_path, "*deform*-downsample*"))
    
    test_files = [os.path.join(original_data_path, f"reg_{f}.nii.gz") for f in og_datasplit['Test']]
    train_files = original_train + generated_train

    split = {'Train': train_files, 'Test': test_files}
    
    print(split)
    
    with open("datasplit_generated.pkl", "wb") as pickle_out:
        pickle.dump(split, pickle_out, protocol=4) # compatible with python 3.6+
        
    
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python create_generated_datasplit.py ./datasplit_70_30.pkl ../registered_niftis/ ../nii_files