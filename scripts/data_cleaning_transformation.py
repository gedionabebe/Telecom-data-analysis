import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scripts.data_extraction import Data_extract

#data = Data_extract()

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
        data[col] = data[col].fillna(data[col].median())

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
