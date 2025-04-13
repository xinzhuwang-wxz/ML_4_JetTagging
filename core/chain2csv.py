import os
import argparse
import uproot
import pandas as pd
import yaml
import numpy as np

'''
 Tchain -> CSV
 param: config
'''

class ReadRoot:
    def __init__(self, file_path, output_path, tree_name, start_entry=None, end_entry=None, cut=None):
        self.file_path = file_path
        self.output_path = output_path
        self.tree_name = tree_name
        self.start_entry = start_entry
        self.end_entry = end_entry
        self.cut = cut

    def to_dataframe(self):
        with uproot.open(self.file_path) as file:
            print(file.keys())  # List all objects in the ROOT file

        tree = uproot.open(self.file_path)[self.tree_name]
        print(tree.num_entries)
        arrays = tree.arrays(entry_start=self.start_entry,
                             entry_stop=self.end_entry,
                             library="np")
        df = pd.DataFrame({key: val for key, val in arrays.items()})

        conditions = [
            df['label_b'] == True,
            df['label_bbar'] == True,
            df['label_c'] == True,
            df['label_cbar'] == True,
            df['label_u'] == True,
            df['label_ubar'] == True,
            df['label_d'] == True,
            df['label_dbar'] == True,
            df['label_s'] == True,
            df['label_sbar'] == True,
            df['label_g'] == True
        ]
        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df['bdtlabel'] = np.select(conditions, choices, default=-1)  # Default to -1 if no condition matches

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tree_name = config.get('tree_name', 'tree')
    start_entry = config.get('start_entry', None)
    end_entry = config.get('end_entry', None)
    cut = config.get('cut', None)

    for tag in ['train', 'val', 'test']:
        root_path = f"data/{tag}.root"
        csv_path = f"data/{tag}.csv"
        reader = ReadRoot(root_path, csv_path, tree_name, start_entry, end_entry, cut)
        reader.to_dataframe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)





