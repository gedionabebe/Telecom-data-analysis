import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from scripts.user_engagement_analysis import k_means
from scripts.experience_analytics import experience_metrics
from scripts.user_engagement_analysis import session_frequency, session_duration, session_total_traffic
from scripts.data_extraction import Data_extract





def experience_score(data):

    label_encode = LabelEncoder()
    exp_metrics = experience_metrics(data)
    exp_metrics['hand_sets'] = label_encode.fit_transform(exp_metrics['hand_sets'])
    exp_cluster = k_means(exp_metrics,3)[1]
    points1 = exp_cluster
    points2 = exp_cluster[exp_cluster['cluster'] == 0]
    euc=euclidean_distances(points1,points2)
    exp_cluster['exp_score'] = np.mean(euc,axis=1)

    return exp_cluster
def engagement_score(data):
    session_freq = session_frequency(data).to_list()
    session_dur = session_duration(data)['Dur. (ms)'].to_list()
    session_data = session_total_traffic(data)['total_data'].to_list()
    user_id = session_total_traffic(data)['MSISDN/Number'].to_list()
    eng_metrics = pd.DataFrame(list(zip(user_id,session_freq,session_dur,session_data)),columns=['MSISDN/Number','session_freq','session_dur','session_data'])
    eng_cluster = k_means(eng_metrics,3)[1]
    points1 = eng_cluster
    points2 = eng_cluster[eng_cluster['cluster'] == 2]
    euc=euclidean_distances(points1,points2)
    eng_cluster['eng_score'] = np.mean(euc,axis=1)

    return eng_cluster

def satisfaction_score(experience_score,engagement_score):

    exp_score = experience_score['exp_score'].to_list()
    eng_score = engagement_score['eng_score'].to_list()
    user_id = experience_score['MSISDN/Number'].to_list()
    satisfaction_metrics = pd.DataFrame(list(zip(user_id,exp_score,eng_score)),columns=['MSISDN/Number','exp_score','eng_score'])
    satisfaction_metrics['satisfaction_score'] = satisfaction_metrics[['exp_score','eng_score']].mean(axis=1)

    return satisfaction_metrics






    

    
