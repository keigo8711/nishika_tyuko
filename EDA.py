####################################################################
    # Combine files, Comvert category variables, Compensate null
####################################################################

import numpy as np
import pandas as pd
from MyPackege import cleaning


if __name__ == '__main__':
    path_to_get = '/Users/keigo/github/nishika/tyuko_mansyon/data/train'
    path_to_save = '/Users/keigo/github/nishika/tyuko_mansyon/data'
    label_encoded_columns = ['都道府県名', '市区町村名', '地区名', '最寄駅：名称', '間取り', '都市計画', '改装', '取引の事情等']

    Preprocessor = cleaning.Preprocessor(before_path=path_to_get, after_path=path_to_save)
    Preprocessor.encode_label(columns=label_encoded_columns)

    arranged_df = Preprocessor.return_df()
    # Preprocessor.save_df()
    # arranged_df = Preprocessor.recall_df()

    df = arranged_df.head(10)
    print(df)
