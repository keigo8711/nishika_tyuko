####################################################################
    # Combine files, Comvert category variables, Compensate null
####################################################################

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from MyPackege import cleaning
# from statsmodels.stats.outliers_influence import variance_inflation_factor



if __name__ == '__main__':
    path_to_get_train = '/Users/keigo/github/nishika/tyuko_mansyon/data/train'
    path_to_get_test = '/Users/keigo/github/nishika/tyuko_mansyon/data/test.csv'
    path_to_save = '/Users/keigo/github/nishika/tyuko_mansyon/data'
    path_for_result = '/Users/keigo/github/nishika/tyuko_mansyon/result'
    label_encoded_columns = ['種類', '都道府県名', '市区町村名', '地区名', '最寄駅：名称', '間取り', '建物の構造', '用途', '今後の利用目的', '都市計画', '改装', '取引の事情等']
    standardized_columns = ['最寄駅：距離（分）', '面積（㎡）', '建築年', '建ぺい率（％）', '容積率（％）']

    Preprocessor = cleaning.Preprocessor(before_path=path_to_get_train, after_path=path_to_save)
    Preprocessor.combine_test_data(test_path=path_to_get_test)

    # # In case we need reprocess data
    Preprocessor.transform_moyori()
    Preprocessor.transform_kenchiku()
    Preprocessor.transform_torihiki()
    Preprocessor.zero_padding()
    Preprocessor.encode_label(columns=label_encoded_columns)

    # For debug
    arranged_df = Preprocessor.return_df()
    temp = arranged_df.head(10)

    # Call data & confirm number of null
    Preprocessor.fill_null_advance(method='median', target_col=['最寄駅：距離（分）', '建ぺい率（％）', '容積率（％）'], referred_col=['面積（㎡）', '建築年'])
    Preprocessor.english_columns()
    arranged_df = Preprocessor.return_df()
    print(arranged_df.isnull().sum())
    Preprocessor.save_df()

    # Visualize and save correlation matrix
    arranged_df = arranged_df[arranged_df['train/test'] == 'train']
    df = arranged_df.astype('float64', errors='ignore')
    df = df.select_dtypes(include='float64')
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,14))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True)
    plt.savefig(os.path.join(path_for_result, 'EDA_result1.png'))

    # # Calculate VIF
    # vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    # vif["features"] = df.columns
    # print(vif)