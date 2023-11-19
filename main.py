####################################################################
    # Build prediction model
####################################################################

from MyPackege import cleaning, modeling


if __name__ == '__main__':
    data_path = '/Users/keigo/github/nishika/tyuko_mansyon/data'
    path_for_result = '/Users/keigo/github/nishika/tyuko_mansyon/result'
    Preprocessor = cleaning.Distributer(get_path=data_path)
    df = Preprocessor.recall_df()
    print(df.head(10))

    cols = ['Municipalities code','Prefecture name','Municipalities name','District name',\
            'Nearest station name','Nearest station minutes','Floor plan','Area','Build year',\
            'Structure of building','Use case','Future purpose','Urban planning','Building coverage ratio',\
            'Floor-area ratio','Renewal','Memo','取引年','取引四半期']
    X_train = df[df['train/test'] == 'train']
    y_train = X_train['Sales value (log)']
    X_test = df[df['train/test'] == 'test']

    X_train = X_train[cols]
    X_test = X_test[cols]

    print('Start modeling')
    Builder = modeling.Model_builder(X_train=X_train, y_train=y_train, X_test=X_test, path_for_result=path_for_result)
    Builder.standardize()
    Builder.holdout_modeling(
        units=[32,128,128],
        dropout_ratios=[0.1,0.1,0.1],
        patience=10,
        epochs=30,
        batch_size=16,
        split_ratio=0.2
    )
    Builder.save_history()
    Builder.show_train_result()
    Builder.save_test_result