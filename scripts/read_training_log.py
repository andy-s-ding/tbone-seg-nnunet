import pandas as pd, re
import numpy as np
import os
import sys

def main(argv):
    log_path = argv[0]
    log_file = open(log_path, "r")
    string = log_file.read()

    rx_epoch = re.compile(r'''
        ^epoch:\s*(?P<epoch_num>.+)
        (?P<epoch_content>[\s\S]+?)
        (?=^epoch|\Z)
    ''', re.MULTILINE | re.VERBOSE)

    rx_train_loss = re.compile(r'''
        ^.*train\s*loss\s*:\s*(?P<train_loss>.+)
    ''', re.MULTILINE | re.VERBOSE)

    rx_val_loss = re.compile(r'''
        ^.*validation\s*loss\s*:\s*(?P<val_loss>.+)
    ''', re.MULTILINE | re.VERBOSE)

    rx_dice_loss = re.compile(r'''
        ^.*\[(?P<dice_loss>.+)\]
    ''', re.MULTILINE | re.VERBOSE)


    result = ((epoch.group('epoch_num'), train.group('train_loss'), val.group('val_loss'), np.mean([float(score) for score in dice.group('dice_loss').split(",")]))
        for epoch in rx_epoch.finditer(string)
        for train in rx_train_loss.finditer(epoch.group('epoch_content'))
        for val in rx_val_loss.finditer(epoch.group('epoch_content'))
        for dice in rx_dice_loss.finditer(epoch.group('epoch_content'))
    )

    df = pd.DataFrame(result, columns = ['Epoch', 'Train Loss', 'Val Loss', 'Dice Score'])
    
    
    if len(argv) == 2:
        out_path = argv[1]
        df.to_csv(out_path)
    else:
        print("No output path specified. Saving to same directory as input file")
        df.to_csv(os.path.splitext(log_path)[0] + ".csv")
    print(df)

if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python read_training_log.py <log_path> <out_path (optional)>
