#####################################
    # Combine files
#####################################

import os
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    
    def encode_label(self, columns: list):
        # Label encoding
        for ii in range(len(columns)):
            col = columns[ii]
            sex_le = LabelEncoder()
            self.df[col] = sex_le.fit_transform(self.df[col])