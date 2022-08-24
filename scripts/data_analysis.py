import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def data_info(data):

    info = data.info()
    columns = data.columns.tolist()

    return info, columns

def percent_missing(data):

    
    totalCells = np.product(data.shape)

    missingCount = data.isnull().sum()

    
    totalMissing = missingCount.sum()

    
    print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")

def handset_manufacturer(data):
    top = data['Handset Manufacturer'].value_counts().head(n=3)

    return top

def top_handsets(data,comp):
    top = data[data['Handset Manufacturer'] == comp]['Handset Type'].value_counts().head(n=5)

    return top
    
def total_ul_per_user(data):
    total = data.groupby('MSISDN/Number').agg({'Total UL (Bytes)':'sum'}).reset_index()

    return total

def total_dl_per_user(data):
    total = data.groupby('MSISDN/Number').agg({'Total DL (Bytes)':'sum'}).reset_index()

    return total

def total_data_vol(data):
    total = data.groupby('MSISDN/Number').agg({'Total DL (Bytes)': 'sum', 'Total UL (Bytes)': 'sum'}).sum(axis=1)

    return total

def total_dur_per_user(data):
    total = data.groupby('MSISDN/Number').agg({'Dur. (ms)':'sum'}).reset_index()

    return total

def mean_value(data):
    num = data.select_dtypes(exclude=['object','bool'])
    num_list = num.columns.tolist()
    mean = data[num_list].mean()

    return mean

def median_value(data):
    num = data.select_dtypes(exclude=['object','bool'])
    num_list = num.columns.tolist()
    median = data[num_list].median()

    return median

def var_value(data):
    num = data.select_dtypes(exclude=['object','bool'])
    num_list = num.columns.tolist()
    var = data[num_list].var()

    return var

def std_value(data):
    num = data.select_dtypes(exclude=['object','bool'])
    num_list = num.columns.tolist()
    std = data[num_list].std()

    return std

def decile_value(data):
    dec_data = data['dec']=pd.qcut(data['Dur. (ms)'],10,labels=False,duplicates='drop')

    return dec_data

def plot_hist(df:pd.DataFrame, column:str, color:str)->None:
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def plot_count(df:pd.DataFrame, column:str) -> None:
    plt.figure(figsize=(40, 10))
    sns.countplot(data=df, x=column)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()

def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='tab20',center=0 )
    plt.title(title, size=18, fontweight='bold')
    plt.show()

def plot_box(df:pd.DataFrame, x_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()

def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

def correlation_matrix(data):
    correlation = data.corr()
    return correlation

def dem_reduction(data):
    pca = PCA()
    x_pca = pca.fit_transform(data)
    x_pca = pd.DataFrame(x_pca) 
    explained_variance = pca.explained_variance_ratio_
    

    return x_pca, explained_variance