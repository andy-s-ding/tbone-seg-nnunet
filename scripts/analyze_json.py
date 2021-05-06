import pandas as pd
import json
import sys
import os

def main(argv):
    
    json_path = argv[0]
    # Write to out_path if given, else write to wherever json path is.
    if len(argv) > 1:    
        out_path = argv[1]
    else:
        out_path = os.path.dirname(json_path)
    
    # Load json
    with open(json_path) as f:
        data = json.load(f)
    
    # Load individual results
    ind_results = (data['results']['all'])
    col_names = (['reference', 'test'] + [f"{i}" for i in range(1, 17)])
    b = pd.DataFrame(columns=col_names)

    # Get dice scores!
    for ind in ind_results:
        a = ind
        c = dict.fromkeys(col_names)
        for label in a:
            try:
    #             print(a[f"{label}"]['Dice'])
                c[label] = a[f"{label}"]['Dice']
            except TypeError:
                # two path names in dict, ignore them
    #             print(a[label])
                c[label] = a[label]
        b = b.append(c, ignore_index=True)

    b.to_csv(os.path.join(out_path, "results.csv"), index=False)
        
    
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python analyze_json.py <json_path> <out_path (optional)>
