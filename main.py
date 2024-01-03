####################################################################
    # Build prediction model
####################################################################

from MyPackege import cleaning, modeling


if __name__ == '__main__':
    print('debug start')
    data_path = '/Users/keigo/github/nishika/tyuko_mansyon/data'
    path_for_result = '/Users/keigo/github/nishika/tyuko_mansyon/result'
    Preprocessor = cleaning.Distributer(get_path=data_path)
    df = Preprocessor.recall_df()
    print(df.head(10))

    # cols = ['ID','Municipalities code','Prefecture name','Municipalities name','District name',\
    #         'Nearest station name','Nearest station minutes','Floor plan','Area','Build year',\
    #         'Structure of building','Use case','Future purpose','Urban planning','Building coverage ratio',\
    #         'Floor-area ratio','Renewal','Memo','取引年','取引四半期']
    cols = ['ID','Municipalities code','Prefecture name','Municipalities name','District name',\
            'Nearest station name','Floor plan','Area','Build year',\
            'Structure of building','Use case','Future purpose','Urban planning',\
            'Renewal','Memo','取引年','取引四半期']
    # ex_cols = ['Municipalities code','Prefecture name','Municipalities name','District name','Floor plan',\
    #         'Structure of building','Use case','Future purpose','Urban planning',\
    #         'Renewal','Memo','取引四半期']
    ex_cols = 'None'
    X_train = df[df['train/test'] == 'train']
    y_train = X_train['Sales value (log)']
    X_test = df[df['train/test'] == 'test']
    IDs = X_test.ID
    X_train = X_train[cols]
    X_test = X_test[cols]

    # Confirm na
    print(X_train.isnull().sum())
    print(X_test.isnull().sum())

    # Drop na
    index = X_train.isnull().any(axis=1)
    X_train = X_train[index == False]
    y_train = y_train[index == False]

    # Confirm na
    print(X_train.isnull().sum())
    print(X_test.isnull().sum())

    print('Start modeling')
    Builder = modeling.Model_builder(X_train=X_train, y_train=y_train, X_test=X_test, path_for_result=path_for_result)
    Builder.standardize(except_col=ex_cols)
    Builder.holdout_modeling(
        units=[64,64,16],
        dropout_ratios=[0.3,0,0],
        # units=[256,1024,1024,256],
        # dropout_ratios=[0.25,0.25,0.25,0.5],
        patience=5,
        epochs=30,
        batch_size=16,
        split_ratio=0.2
    )
    Builder.save_history()
    Builder.show_train_result()
    Builder.save_test_result(id=IDs)