import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def to_numbers(data,col):
    data[col] = pd.to_numeric(data[col])
    return data

def to_datatime(data,col):
    data[col] = pd.to_datetime(data[col])
    return data

def drop_duplicate(data):
    data.drop_duplicates(inplace = True)
    return data

def fill_missing_values(data,col):
    
    if data.dtypes[col] == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])

        return data

    elif data.dtypes[col] != 'object':
        data[col] = data[col].fillna(data[col].mean())

        return data

def fix_outlier(data, column):
    data[column] = np.where(data[column] > data[column].quantile(0.95), data[column].median(),data[column])
    
    return data

def convert_bytes_to_megabytes(data, col):

    megabyte = 1*10e+5
    data[col] = data[col] / megabyte
    
    return data[col]

def standard_scaler(data):
    stand_scaler = StandardScaler()
    scaled_data = stand_scaler.fit_transform(data)

    return scaled_data

def normalizer(data):
  minmax_scaler = MinMaxScaler()
  normalized_data = minmax_scaler.fit_transform(data)

  return normalized_data

def convert_to_datetime( df, col_name):   
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
           
        return df
    
def convert_to_integer(df, col_name):
        
        df[col_name] = df[col_name].astype("int64")
            
        return df
    
def convert_to_string(df, col_name):
        
        df[col_name] = df[col_name].astype("string")
            
        return df

def drop_duplicate(df):
        
        df = df.drop_duplicates()
            
        return df

def drop_column(df, col_name):
        
        df.drop([col_name], axis=1, inplace=True)
            
        return df

def get_missing_values(df):
        
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})

        missing_value_df.sort_values('percent_missing', inplace=True)
            
        return missing_value_df

def fix_missing_ffill(df, col_name):
        
        df[col_name] = df[col_name].fillna(method='ffill')
            
        return df

def fix_missing_bfill(df, col_name):
        
        df[col_name] = df[col_name].fillna(method='bfill')
            
        return df

def fix_missing_value(df, col_name, value):
       
        df[col_name] = df[col_name].fillna(value)
            
        return df

def fix_missing_median(df, col_name):
        
        df[col_name] = df[col_name].fillna(df[col_name].median())
            
        return df

def get_row_nan_percentage(df):
        
        rows_with_nan = [index for index,row in df.iterrows() if row.isnull().any()]
        percentage = (len(rows_with_nan) / df.shape[0]) * 100
           
        return percentage

def fix_outliers(df):
    for col in df.select_dtypes('float64').columns.tolist():
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - (IQR * 1.5)
        upper = Q3 + (IQR * 1.5)

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])
               
        return df
