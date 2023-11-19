#####################################
    # Combine files
#####################################

import os
import glob
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


def wareki2seireki(wareki_s, def_value=0):
    era_dic = {"明治": 1868, "大正": 1912, "昭和": 1926, "平成": 1989, "令和": 2019}
    s = re.match(r'(明治|大正|昭和|平成|令和)([0-9]+|元)年', str(wareki_s))
    if s is None: return def_value
    y = int(s.group(2)) if s.group(2) != '元' else 1
    return era_dic[s.group(1)] + y - 1


class Preprocessor:

    def __init__(self, before_path: str, after_path: str):
        # Fix paths to recieve anr return
        self.get_path = before_path
        self.return_path = after_path

        if self.get_path != None:
            # Combine Files
            print('-- Start pre-processing --')
            file_list = glob.glob(os.path.join(self.get_path, "*.csv"))
            for ii, file_path in enumerate(file_list):
                if ii == 0:
                    self.df = pd.read_csv(file_path)  # Initialize
                else:
                    self.df = pd.concat([self.df, pd.read_csv(file_path)])  # Append
            self.df.replace("", np.nan, inplace=True)
            self.df = self.df.reset_index(drop=True)

    def combine_test_data(self, test_path: str):
        test_df = pd.read_csv(test_path)
        self.df['train/test'] = 'train'
        test_df['train/test'] = 'test'
        self.df = pd.concat([self.df, test_df])  # Append
        self.df.replace("", np.nan, inplace=True)
        self.df = self.df.reset_index(drop=True)

    def return_df(self) -> pd.DataFrame:
        print('-- End pre-processing --')
        return self.df

    def save_df(self):
        path = self.return_path + '/saved_data.csv'
        self.df.to_csv(path)
        print('-- End pre-processing --')

    def recall_df(self) -> pd.DataFrame:
        path = self.return_path + '/saved_data.csv'
        self.df = pd.read_csv(path)
        return self.df

    def encode_label(self, columns: list):
        # Label encoding
        for ii in range(len(columns)):
            col = columns[ii]
            sex_le = LabelEncoder()
            self.df[col] = sex_le.fit_transform(self.df[col])

    def transform_moyori(self):
        # apply transformation for each
        for ii in self.df.index:
            if ii == int(len(self.df.index)/2):
                print('- Half time -')

            if self.df.loc[ii,'最寄駅：距離（分）'] == '30分?60分':
                self.df.loc[ii,'最寄駅：距離（分）'] = 45
            elif self.df.loc[ii,'最寄駅：距離（分）'] == '1H?1H30':
                self.df.loc[ii,'最寄駅：距離（分）'] = 75
            elif self.df.loc[ii,'最寄駅：距離（分）'] == '1H30?2H':
                self.df.loc[ii,'最寄駅：距離（分）'] = 105
            elif self.df.loc[ii,'最寄駅：距離（分）'] == '2H?':
                self.df.loc[ii,'最寄駅：距離（分）'] = 120

    def transform_kenchiku(self):
        # apply transformation for each
        for ii in self.df.index:
            if ii == int(len(self.df.index)/2):
                print('- Half time -')

            if self.df.loc[ii,'建築年'] == '戦前':
                self.df.loc[ii,'建築年'] = 1945
            elif self.df.loc[ii,'建築年'] != np.nan:
                self.df.loc[ii,'建築年'] = wareki2seireki(self.df.loc[ii,'建築年'])

    def transform_torihiki(self):
        # apply transformation for each
        self.df['取引年'] = np.nan
        self.df['取引四半期'] = np.nan
        for ii in self.df.index:
            if ii == int(len(self.df.index)/2):
                print('- Half time -')

            if self.df.loc[ii,'取引時点'] != np.nan:
                string = self.df.loc[ii,'取引時点']
                self.df.loc[ii,'取引年'] = int(string[:4])
                self.df.loc[ii,'取引四半期'] = int(string[6])

    def zero_padding(self):
        self.df = self.df.fillna({'用途': '0', '今後の利用目的': '0'})

    def fill_null_basic(self):
        self.df.fillna(self.df.median(numeric_only=True))

    def fill_null_advance(self, method: str, target_col: list, referred_col: list):
        method_to_fill = ['median', 'mean']
        for ii in range(len(target_col)):
            if method == method_to_fill[0]:
                print('Null is filled by ' + str(method_to_fill[0]))
                self.df[target_col[ii]]= self.df.groupby(referred_col)[target_col[ii]].apply(lambda row : row.fillna(row.median()))
            elif method == method_to_fill[1]:
                print('Null is filled by ' + str(method_to_fill[1]))
                self.df[target_col[ii]]= self.df.groupby(referred_col)[target_col[ii]].apply(lambda row : row.fillna(row.mean()))
            else:
                raise ValueError("Select your method in " + str(method_to_fill))

    def english_columns(self):
        column_mapping = {'種類': 'Type',
                          '地域': 'Region',
                          '市区町村コード': 'Municipalities code',
                          '都道府県名': 'Prefecture name',
                          '市区町村名': 'Municipalities name',
                          '地区名': 'District name',
                          '最寄駅：名称': 'Nearest station name',
                          '最寄駅：距離（分）': 'Nearest station minutes',
                          '間取り' : 'Floor plan',
                          '面積（㎡）': 'Area',
                          '土地の形状': 'Shape of land',
                          '間口': 'Frontage',
                          '延床面積（㎡）': 'Total floor area',
                          '建築年': 'Build year',
                          '建物の構造': 'Structure of building',
                          '用途': 'Use case',
                          '今後の利用目的': 'Future purpose',
                          '前面道路：方位': 'Road direction',
                          '前面道路：種類': 'Road type',
                          '前面道路：幅員（ｍ）': 'Road width',
                          '都市計画': 'Urban planning',
                          '建ぺい率（％）': 'Building coverage ratio',
                          '容積率（％）': 'Floor-area ratio',
                          '取引時点': 'Transaction point',
                          '改装': 'Renewal',
                          '取引の事情等': 'Memo',
                          '取引価格（総額）_log': 'Sales value (log)'}
        self.df.rename(columns=column_mapping, inplace=True)


class Distributer(Preprocessor):

    def __init__(self, get_path: str):
        super().__init__(before_path=None, after_path=get_path)