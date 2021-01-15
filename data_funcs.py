import numpy as np
import pandas as pd


def read_data(file):
    df = pd.read_csv(file)
    return df


def step_1(df):
    # One-Story Residences include classification codes 202, 203, 204
    ONE_STORY_LIST = [202, 203, 204]
    one_story_index = df['Property_Class'].isin(ONE_STORY_LIST)
    # Drop rows with missing values in Half_Baths and Full_Baths
    df = (df.loc[one_story_index, :]
            .dropna(how='any', subset=['Full_Baths', 'Half_Baths']))
    return df


def step_2(df):
    # Create column Total_Baths
    # Create the column Log_Sale_Price
    # Drop columns Full_Baths and Half_Baths
    df = (df.assign(Total_Baths=lambda x: x['Full_Baths'] + (x['Half_Baths'] * .5),
                    Log_Sale_Price=lambda x: np.log(x['Sale_Price']))
            .drop(columns=['Full_Baths', 'Half_Baths']))
    return df


def step_3(df):

    # Recode the "Basement" column to "Full" or "Partial". Otherwise, "None"
    BASEMENT_DICT = {'Full basement': 'Full',
                     'Partial basement': 'Partial',
                     'Slab basement': 'Partial',
                     'crawlspace': 'Partial'}
    df = (df.assign(Basement=lambda x: (x['Basement'].map(BASEMENT_DICT)
                                                     .fillna('None'))))
    return df


def step_4(df):
    # Set its index to Tax_Year
    # Set the datatype for Central_Air to category
    df = (df.set_index('Tax_Year', drop=True)
            .assign(Central_Air=lambda x: x['Central_Air'].astype('category')))
    return df