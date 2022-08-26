import pandas as pd
from sklearn.cluster import KMeans


def session_frequency(data):

    session_freq = data['MSISDN/Number'].value_counts()

    return session_freq

def session_duration(data):

    session_dur = pd.DataFrame( data.groupby('MSISDN/Number')['Dur. (ms)'].first().reset_index())

    return session_dur

def session_total_traffic(data):
    
    total_data = data.groupby('MSISDN/Number').agg({'Total DL (Bytes)': 'sum', 'Total UL (Bytes)': 'sum'}).sum(axis=1).tolist()
    user_id = data.groupby('MSISDN/Number').agg({'Total DL (Bytes)': 'sum', 'Total UL (Bytes)': 'sum'}).sum(axis=1).index[:]
    total_session_data = pd.DataFrame(list(zip(user_id,total_data)),columns=['MSISDN/Number','total_data'])

    return total_session_data

def total_app_traffic(data,dl_col,ul_col,new_col):

    user_id = data.groupby('MSISDN/Number').agg({'Total DL (Bytes)': 'sum', 'Total UL (Bytes)': 'sum'}).sum(axis=1).index[:]
    total_app_data = data.groupby('MSISDN/Number').agg({dl_col: 'sum', ul_col: 'sum'}).sum(axis=1).tolist()
    total = pd.DataFrame(list(zip(user_id,total_app_data)),columns=['MSISDN/Number',new_col])

    return total

def k_means(data,cluster):
    km = KMeans(
    n_clusters=cluster, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0)

    output = km.fit_predict(data)
    data['cluster'] = output
    return output, data