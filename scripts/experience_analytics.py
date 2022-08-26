import pandas as pd
from experience_analytics import k_means
from sklearn.preprocessing import LabelEncoder

def experience_metrics(data):

    avg_tcp = data.groupby('MSISDN/Number').agg({'TCP DL Retrans. Vol (Bytes)': 'mean', 'TCP UL Retrans. Vol (Bytes)': 'mean'}).sum(axis=1)
    avg_rtt = data.groupby('MSISDN/Number').agg({'Avg RTT DL (ms)': 'mean', 'Avg RTT UL (ms)': 'mean'}).sum(axis=1)
    hand_sets = data.groupby('MSISDN/Number')['Handset Type'].first()
    avg_throughput = data.groupby('MSISDN/Number').agg({'Avg Bearer TP DL (kbps)': 'mean', 'Avg Bearer TP UL (kbps)': 'mean'}).sum(axis=1)
    user_id = data.groupby('MSISDN/Number')['MSISDN/Number'].first()
    exp_metrics = pd.DataFrame(list(zip(user_id,avg_tcp,avg_rtt,hand_sets,avg_throughput)),columns=['MSISDN/Number','avg_tcp','avg_rtt','hand_sets','avg_throughput'])

    return exp_metrics

def metrics_analysis(data,metrics):

    top_10 = data.sort_values(by=metrics,ascending=False).head(n=10)
    bottom_10 = data.sort_values(by=metrics,ascending=True).head(n=10)
    most_frequent= data[metrics].mode()

    return top_10, bottom_10, most_frequent

def metrics_per_handset(data):

    throughput = data.groupby('hand_sets')['avg_throughput'].first()
    tcp = data.groupby('hand_sets')['avg_tcp'].first()

    return throughput, tcp

def metrics_cluster(data,cluster):

    label_encode = LabelEncoder()
    data['hand_sets'] = label_encode.fit_transform(data['hand_sets'])
    output, data = k_means(data,cluster)

    return output, data



