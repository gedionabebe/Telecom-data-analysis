import pandas as pd
import os,sys

#sys.path.append(os.path.abspath(os.path.join("../..")))
#sys.path.append('C:/Users/User/Desktop/Telecom-data-analysis/')
#sys.path.insert(0, 'C:/Users/User/Desktop/Telecom-data-analysis/data/')


def Data_extract():
    data = pd.read_csv("C:/Users/User/Desktop/Telecom-data-analysis/data/Week1_challenge_data_source(CSV).csv", na_values=['?',None])
    return data

