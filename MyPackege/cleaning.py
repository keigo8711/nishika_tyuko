#####################################
    # Combine files
#####################################

import os
import glob
import pandas as pd


class Preprocessor:

    def __init__(self, before_path: str, after_path: str):
        # Fix paths to recieve anr return
        self.get_path = before_path
        self.return_path = after_path

        # Combine Files
        print('-- Start pre-processing --')
        file_list = glob.glob(os.path.join(self.get_path, "*.csv"))
        for ii, file_path in enumerate(file_list):
            if ii == 0:
                self.df = pd.read_csv(file_path)  # Initialize
            else:
                self.df = pd.concat([self.df, pd.read_csv(file_path)])  # Append

    def return_df(self) -> pd.DataFrame:
        print('-- End pre-processing --')
        return self.df
    
    def save_df(self):
        path = self.return_path + '/saved_data.csv'
        self.df.to_csv(path)
        print('-- End pre-processing --')
    
    def recall_df(self) -> pd.DataFrame:
        path = self.return_path + '/saved_data.csv'
        return pd.read_csv(path)