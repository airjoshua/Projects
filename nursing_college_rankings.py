import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale


def rename_columns(df):
    col_renames = \
        {'Graduation Rate (12%)': 'Graduation_Rate',
         'Student-Faculty Ratio (12%)': 'Student_Faculty_Ratio',
         'Net Price (12%)': 'Net_Price',
         'Cohort Year *': 'Cohort_Year',
         'Cohort Default Rates (12%)': 'Cohort_Default_Rates',
         'All Undergraduates, % Receiving Federal Student Loans (12%)': 'Undergrads_Receiving_Fed_Stu_Loans_Pct',
         'NCLEX PASS Rate (40%)': 'NCLEX_PASS_Rate',
         'Student Population': 'Student_Population',
         'Undergraduate Students': 'Undergraduate_Students'}
    return df.rename(columns=col_renames)


def to_numeric(df):
    to_numeric_cols = ['Undergraduate_Students', 'Cohort_Default_Rates', 'Net_Price',
                       'Undergrads_Receiving_Fed_Stu_Loans_Pct',
                       'Student_Population', 'Graduation_Rate', 'NCLEX_PASS_Rate']
    df.loc[:, to_numeric_cols] = df.loc[:, to_numeric_cols].apply(pd.to_numeric)
    return df


def replace_values(df):
    values_dict = \
        {'Net_Price': {'\$': '',
                       ',': ''},
         'Undergraduate_Students': {',': ''},
         'Cohort_Default_Rates': {',': ''},
         'Undergrads_Receiving_Fed_Stu_Loans_Pct': {'%': ''},
         'Graduation_Rate': {'%': ''},
         'NCLEX_PASS_Rate': {',': ''}}
    return df.replace(values_dict, regex=True)


def data_cleanup(df):
    df = \
        (df.pipe(rename_columns)
           .pipe(replace_values)
           .pipe(to_numeric)
           .assign(School_Type=lambda x: np.where(x['Type'].str.contains('Public', na=False), 'Public',
                                                  np.where(x['Type'].str.contains('Private', na=False), 'Private', None)),
                   Student_Faculty_Ratio=lambda x: pd.to_numeric((x['Student_Faculty_Ratio'].str.split(':')
                                                                                            .str.get(0))),
                   Graduation_Rate=lambda x: x['Graduation_Rate'].divide(100))
           .assign(School_Type=lambda x: x['School_Type'].fillna('Public'))
           .dropna(subset=['NCLEX_PASS_Rate'])
           .drop('Unnamed: 23', axis=1))
    return df



def dollar_signs(df):
    price_20 = np.percentile(df['Net_Price'], 20)
    price_40 = np.percentile(df['Net_Price'], 40)
    price_60 = np.percentile(df['Net_Price'], 60)
    price_80 = np.percentile(df['Net_Price'], 80)

    conditions = [(df['Net_Price'] <= price_20),
                  (df['Net_Price'].gt(price_20) & df['Net_Price'].le(price_40)),
                  (df['Net_Price'].gt(price_40) & df['Net_Price'].le(price_60)),
                  (df['Net_Price'].gt(price_60) & df['Net_Price'].le(price_80)),
                  df['Net_Price'].gt(price_80)]
    values = ['$', '$$', '$$$', '$$$$', '$$$$$']

    df = df.assign(Dollar_Signs=np.select(conditions, values))
    return df


def group_by_region_school_type(df, select_col):
    group_by_cols = ['Region', 'School_Type']
    df = (df.groupby(group_by_cols)[select_col]
            .median())
    return df


def impute_missing_data(df):
    median_cohort_default = group_by_region_school_type(df, 'Cohort_Default_Rates')
    median_grad_rate = group_by_region_school_type(df, 'Graduation_Rate')
    median_net_price = group_by_region_school_type(df, 'Net_Price')
    median_Student_Faculty_Ratio = group_by_region_school_type(df, 'Student_Faculty_Ratio')
    median_student_population = group_by_region_school_type(df, 'Student_Population')
    median_nclex = group_by_region_school_type(df, 'NCLEX_PASS_Rate')
    df = \
        (df.assign(Cohort_Default_Rates=lambda x: np.where(x['Cohort_Default_Rates'].isnull(),
                                                           x.set_index(['Region', 'School_Type']).index.map(median_cohort_default),
                                                           x['Cohort_Default_Rates']),
                   Graduation_Rate=lambda x: np.where(x['Graduation_Rate'].isnull(),
                                                      x.set_index(['Region', 'School_Type']).index.map(median_grad_rate),
                                                      x['Graduation_Rate']),
                   Net_Price=lambda x: np.where(x['Net_Price'].isnull(),
                                                x.set_index(['Region', 'School_Type']).index.map(median_net_price),
                                                x['Net_Price']),
                   Student_Faculty_Ratio= lambda x: np.where(x['Student_Faculty_Ratio'].isnull(),
                                                             x.set_index(['Region', 'School_Type']).index.map(median_Student_Faculty_Ratio),
                                                             x['Student_Faculty_Ratio']),
                   Student_Population=lambda x: np.where(x['Student_Population'].isnull(),
                                                            x.set_index(['Region', 'School_Type']).index.map(median_student_population),
                                                            x['Student_Population']),
                   NCLEX_PASS_Rate=lambda x: np.where(x['NCLEX_PASS_Rate'].isnull(),
                                                         x.set_index(['Region', 'School_Type']).index.map(median_nclex),
                                                         x['NCLEX_PASS_Rate'])
                   ))
    return df


def rankings(df):
    df.loc[:, 'Score'] = \
        (25 * (minmax_scale(df['Graduation_Rate'], feature_range=(0, .80)) +
               minmax_scale(df['Graduation_Rate'] * df['Student_Population'], feature_range=(0, .20)))
         +
         25 * (minmax_scale((1 - df['Cohort_Default_Rates']), feature_range=(0, .90))
               +
               minmax_scale(((1 - df['Cohort_Default_Rates']) * df['Student_Population']),
                            feature_range=(0, .10)))
         +
            10 * (minmax_scale((1 / np.log(df['Net_Price']))))
         +
            10 * (minmax_scale((1 / np.log(df['Student_Faculty_Ratio']))))
         +
            30 * (minmax_scale(df['NCLEX_PASS_Rate'], feature_range=(0, .80))
                  +
                  minmax_scale(df['NCLEX_PASS_Rate'] * df['Student_Population'], feature_range=(0, .20)))
         )
    df = (df.sort_values(by='Score', ascending=False)
            .assign(School_Rankings=range(1, df.shape[0] + 1),
                    Score=lambda x: minmax_scale(x['Score'], feature_range=(50, 99)))
          )
    return df


def pre_processing(df):
    return (df.pipe(data_cleanup)
              .pipe(impute_missing_data)
              .pipe(dollar_signs))


def national_rankings(df):
    return df.pipe(rankings)


def regional_rankings(df):
    for region in df['Region'].unique():
        region_df = df.query(f'Region == "{region}"')
        (region_df.pipe(rankings)
                  .to_csv(f'{region}_Rankings.csv', index=False))


def main():
    schools = (pd.read_excel('Rankings/data.xlsx')
               .pipe(data_cleanup)
               .pipe(impute_missing_data)
               .pipe(dollar_signs))
    national_rankings(schools)
    regional_rankings(schools)


if __name__ == "__main__":
    main()