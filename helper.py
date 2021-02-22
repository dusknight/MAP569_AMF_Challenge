import pandas as pd
import numpy as np


def read_x_train(filename: str, includeTrader=True, includeShare=True, includeDay=True) -> pd.DataFrame:
    """Read a csv file and do a little bit conversion from string to int
    Args
        filename: csv file name
        includeTrader: if the column 'Trader' is included
        includeShare: if the column 'Share' is included
        includeDay: if the column 'Day' is included

    Returns
        x_remove: pd.DataFrame
            A dataframe after cleaning
    """
    x_remove = pd.read_csv(filename)
    x_remove['Day'] = x_remove['Day'].apply(lambda x: int(x.split('_')[1]))
    # x_remove['Trader'] = x_remove['Trader'].apply(
    #     lambda x: int(x.split('_')[1]))
    x_remove['Share'] = x_remove['Share'].apply(
        lambda x: int(x.split('_')[1]))
    if not includeTrader:
        x_remove = x_remove.drop(
            ['Trader'], axis=1, inplace=False)
    if not includeShare:
        x_remove = x_remove.drop(
            ['Share'], axis=1, inplace=False)
    if not includeDay:
        x_remove = x_remove.drop(
            ['Day'], axis=1, inplace=False)
    x_remove = x_remove.drop(
        ['Index'], axis=1, inplace=False)
    return x_remove


def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Deal with the NaN value in our specific dataFrame. 
       Will simply fill the NaN in OMR by 0. For the NaN concerning the dt,
       will do the conversion x -> 1/x then fill the NaN by 0.
    Args
        dataframe: pd.DataFrame

    Returns
        x_remove: pd.DataFrame
            A dataframe after cleaning
    """
    dt_list = ['min_dt_TV1', 'mean_dt_TV1', 'med_dt_TV1', 'min_dt_TV1_TV2',
               'mean_dt_TV1_TV2', 'med_dt_TV1_TV2', 'min_dt_TV1_TV3',
               'mean_dt_TV1_TV3', 'med_dt_TV1_TV3', 'min_dt_TV1_TV4',
               'mean_dt_TV1_TV4', 'med_dt_TV1_TV4']
    df['OMR'] = df['OMR'].fillna(0)
    df['OTR'] = df['OTR'].fillna(0)
    for col_name in dt_list:
        df[col_name] = df[col_name].apply(
            lambda x: 1/x if pd.notnull(x) else 0)
    return df


# x = read_x_train('data/AMF_train_X.csv')
# print(x)
# x = fill_nan(x)
# print(x)
