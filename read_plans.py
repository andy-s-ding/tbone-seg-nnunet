import pickle
import sys


def main(argv):
    data_path = argv[0]
    plans = pickle.load(open(data_path, 'rb'))
    print(plans)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
