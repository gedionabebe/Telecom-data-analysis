import pandas as pd
import os

data_path = os.path.dirname(os.getcwdb()).join("/data","")


def Data_extract():
    data = pd.read_excel(data_path)
    return data