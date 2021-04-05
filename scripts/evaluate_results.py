import nibabel as nib
import numpy as np
import os
import glob
import sys
import nrrd
import json

def main(argv):
    json_path = argv[0]
    out_path = argv[1]

    with open(json_path) as f:
        data = json.load(f)
    print(data)
    breakpoint()

    page_names = ['Accuracy', 'Dice', 'False Discovery Rate', 'False Negative Rate', 'False Omission Rate',
                    'False Positive Rate', 'Jaccard', 'Negative Predictive Value', 'Precision', 'Recall',
                    'Total Positives Reference', 'Total Positives Test', 'True Negative Rate']
    column_names = [f'i' for i in range(16)]
    result_df = pd.DataFrame(columns = column_names)

    for CT in data['results']['all']:
        for label in range(1, 17):
            CT_dict = {}
            for page_name in page_names:
                CT_dict[page_name] = CT['label'][page_name]
            mapping_df = mapping_df.append(pd.DataFrame(CT_dict))




if __name__ == '__main__':
    main(sys.argv[1:])

    # usage: python evaluate_results.py json_path
    # python evaluate_results.py ../01_jsoong/nnUnet/nnUNet_trained_models/nnUNet/3d_fullres/Task101_TemporalBone/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed/summary.json
