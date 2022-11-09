import sys
import pandas as pd 
import numpy as np
sys.path.insert(0,'../scripts/')
from scripts.data_fetch import get_data
from scripts.data_cleaning_transformation import to_datatime, to_numbers


def test_data_fetch():
    data = get_data('data/Week1_challenge_data_source.csv','C:/Users/User/Desktop/Telecom-data-analysis','Week1_challenge_data_source_v1')
    assert data.shape[0] == 150001
def test_data_cleaning():
    data = pd.DataFrame([['6','4/25/2019 7:36']], columns=['Numbers','Date'])
    data = to_numbers(data,'Numbers')
    data = to_datatime(data,'Date')
    assert data.shape[0] == 1

    