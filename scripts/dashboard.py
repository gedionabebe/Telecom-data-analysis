import numpy as np
import pandas as pd
import streamlit as st
from data_extraction import Data_extract
from data_analysis import plot_hist, plot_heatmap,handset_manufacturer,data_info
from satisfaction_analysis import engagement_score, experience_score, satisfaction_score
from user_engagement_analysis import total_app_traffic
from experience_analytics import experience_metrics, metrics_per_handset
from model import regression_model
import plotly.express as px
import sys, os
sys.path.insert(0,'../scripts/')
from data_fetch import get_data


clean_data_df = get_data('data/cleaned_data.csv','C:/Users/User/Desktop/Telecom-data-analysis','cleaned_data_v1')
user_engagement_data = get_data('data/user_engagement_data.csv','C:/Users/User/Desktop/Telecom-data-analysis','user_engagement_v1')
user_experience_data = get_data('data/user_experience_data.csv','C:/Users/User/Desktop/Telecom-data-analysis','user_experience_v1')
user_satisfaction_score_data = get_data('data/user_satisfaction_score_data.csv','C:/Users/User/Desktop/Telecom-data-analysis','user_satisfaction_score_data_v1')

def user_overview():
    st.title("User Overview Analysis")
    st.header('Overview of our data set')
    st.write(clean_data_df)
    st.header("Top Handsets")
    st.write(clean_data_df['Handset Type'].value_counts())
    fig = px.bar(clean_data_df['Handset Type'].value_counts().rename_axis('Handset Type').reset_index(name='counts').head(10), x='Handset Type', y='counts')
    st.plotly_chart(fig)
    top_handset_manufacturers = clean_data_df['Handset Manufacturer'].value_counts().head(3)
    top_handset_manufacturers = clean_data_df[clean_data_df["Handset Manufacturer"].isin(top_handset_manufacturers.index.tolist())]
    top_handsets = top_handset_manufacturers['Handset Type'].groupby(clean_data_df['Handset Manufacturer']).apply(lambda x: x.value_counts().head(5))
    st.header("Top Handsets by manufactureres")
    st.dataframe(top_handsets)
    st.header("User with the top number of sessions")
    number_of_xdr = clean_data_df.groupby('MSISDN/Number')['MSISDN/Number'].agg('count').reset_index(name='Bearer Id').sort_values(by='Bearer Id', ascending=False)
    number_of_xdr.rename(columns={number_of_xdr.columns[1]: 'number of xDR sessions'}, inplace=True)
    st.dataframe(number_of_xdr.head(10))
    fig = px.bar(number_of_xdr.head(10), x='MSISDN/Number',y='number of xDR sessions')
    fig.update_layout(xaxis_type='category')
    st.plotly_chart(fig)
    st.header("User with the top total duration of sessions")
    sum_duration_of_sessions = clean_data_df.groupby('MSISDN/Number').agg({'Dur. (ms)': 'sum'}).sort_values(by='Dur. (ms)', ascending=False)
    sum_duration_of_sessions.rename(columns={sum_duration_of_sessions.columns[0]: 'duration of xDR sessions (total)'}, inplace=True)
    sum_duration_of_sessions['duration of xDR sessions (total)'] = sum_duration_of_sessions['duration of xDR sessions (total)'].astype("int64")
    st.dataframe(sum_duration_of_sessions.head(10))
    st.header("User with the top avarage duration of sessions")
    avg_duration_of_sessions = clean_data_df.groupby('MSISDN/Number').agg({'Dur. (ms)': 'mean'}).sort_values(by='Dur. (ms)', ascending=False)
    avg_duration_of_sessions.rename(columns={avg_duration_of_sessions.columns[0]: 'duration of xDR sessions (AVG)'}, inplace=True)
    avg_duration_of_sessions
    st.dataframe(avg_duration_of_sessions.head(10))
    st.header("User with the top total data used")
    data_volumes = clean_data_df.groupby('MSISDN/Number')[['Total UL (Bytes)', 'Total DL (Bytes)','Total Data Volume (Bytes)']].sum().sort_values(by='Total Data Volume (Bytes)', ascending=False)
    data_volumes['Total Data Volume (Bytes)'] = data_volumes['Total Data Volume (Bytes)'].astype("int64")
    data_volumes['Total DL (Bytes)'] = data_volumes['Total DL (Bytes)'].astype("int64")
    data_volumes['Total UL (Bytes)'] = data_volumes['Total UL (Bytes)'].astype( "int64")
    st.dataframe(data_volumes)


def user_engagement():
    st.title("User Engagement Analysis")
    st.dataframe(clean_data_df.head(1000))
    app_clean_data_df = clean_data_df[['MSISDN/Number', 'Social Media Data Volume (Bytes)', 'Google Data Volume (Bytes)','Email Data Volume (Bytes)', 'Youtube Data Volume (Bytes)', 'Netflix Data Volume (Bytes)','Gaming Data Volume (Bytes)', 'Other Data Volume (Bytes)']]
    app_clean_data_df = app_clean_data_df.groupby('MSISDN/Number').agg({'Social Media Data Volume (Bytes)': 'sum','Google Data Volume (Bytes)': 'sum','Email Data Volume (Bytes)': 'sum','Youtube Data Volume (Bytes)': 'sum','Netflix Data Volume (Bytes)': 'sum','Gaming Data Volume (Bytes)': 'sum','Other Data Volume (Bytes)': 'sum'})
    clean_data_df = clean_data_df[['MSISDN/Number', 'Bearer Id', 'Dur. (ms)', 'Total Data Volume (Bytes)']]
    clean_data_df = clean_data_df.groupby('MSISDN/Number').agg({'Bearer Id': 'count', 'Dur. (ms)': 'sum', 'Total Data Volume (Bytes)': 'sum'})
    clean_data_df = clean_data_df.rename(columns={'Bearer Id': 'number of xDR Sessions'})
    st.write("")
    st.header('Top 10 Numbers (Users) with highest')
    option = st.selectbox('Top 10 Numbers (Users) with highest',('Number of xDR Sessions', 'Number of Duration', 'Total Data Volume'))

    if option == 'Number of xDR Sessions':
        data = clean_data_df.sort_values('number of xDR Sessions', ascending=False).head(10)
        name = 'number of xDR Sessions'
    elif option == 'Number of Duration':
        data = clean_data_df.sort_values('Dur. (ms)', ascending=False).head(10)
        name = 'Dur (ms)'
    elif option == 'Total Data Volume':
        data = clean_data_df.sort_values('Total Data Volume (Bytes)', ascending=False).head(10)
        name = 'Total Data Volume (Bytes)'
    data = data.reset_index('MSISDN/Number')
    fig = px.pie(data, names='MSISDN/Number', values=name)
    st.plotly_chart(fig)
    st.dataframe(data)
    st.write("")
    st.header('Top 10 Engaged Users Per App')
    app_option = st.selectbox('Top 10 Engaged Users Per App',('Social Media', 'Youtube','Google', 'Email', 'Netflix', 'Gaming', 'Other'))

    if app_option == 'Social Media':
        app_data = app_clean_data_df.sort_values('Social Media Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Social Media Data Volume (Bytes)'
    elif app_option == 'Youtube':
        app_data = app_clean_data_df.sort_values('Youtube Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Youtube Data Volume (Bytes)'
    elif app_option == 'Google':
        app_data = app_clean_data_df.sort_values('Google Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Google Data Volume (Bytes)'
    elif app_option == 'Email':
        app_data = app_clean_data_df.sort_values('Email Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Email Data Volume (Bytes)'
    elif app_option == 'Netflix':
        app_data = app_clean_data_df.sort_values('Netflix Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Netflix Data Volume (Bytes)'
    elif app_option == 'Gaming':
        app_data = app_clean_data_df.sort_values('Gaming Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Gaming Data Volume (Bytes)'
    else:
        app_data = app_clean_data_df.sort_values('Other Data Volume (Bytes)',ascending=False).head(10)
        app_name = 'Other Data Volume (Bytes)'
    app_data = app_data.reset_index('MSISDN/Number')
    app_fig = px.pie(app_data, names='MSISDN/Number', values=app_name)
    st.plotly_chart(app_fig)
    st.dataframe(app_data)
    st.title("User Clusters")
    st.write("")
    st.dataframe(user_engagement_data.head(1000))
    st.write("")
    st.markdown("***Users classified into 6 clusters based on their engagement(i.e. number of xDR sessions, duration and total data used).***")
    fig = px.scatter(user_engagement_data, x='Total Data Volume (Bytes)', y='Dur. (ms)',color='cluster', size='xDR Sessions')
    st.plotly_chart(fig)


def user_experience():
    st.title("User Experience Analysis")
    tellco_exprience_df = clean_data_df[['MSISDN/Number', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)','Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Handset Type']]
    tellco_exprience_df['Total Avg RTT (ms)'] = tellco_exprience_df['Avg RTT DL (ms)'] + \
    tellco_exprience_df['Avg RTT UL (ms)']
    tellco_exprience_df['Total Avg Bearer TP (kbps)'] = tellco_exprience_df['Avg Bearer TP DL (kbps)'] + \
    tellco_exprience_df['Avg Bearer TP UL (kbps)']
    tellco_exprience_df['Total TCP Retrans. Vol (Bytes)'] = tellco_exprience_df['TCP DL Retrans. Vol (Bytes)'] + \
    tellco_exprience_df['TCP UL Retrans. Vol (Bytes)']
    tellco_exprience_df = tellco_exprience_df[['MSISDN/Number', 'Total Avg RTT (ms)', 'Total Avg Bearer TP (kbps)', 'Total TCP Retrans. Vol (Bytes)', 'Handset Type']]
    tellco_exprience_df1 = tellco_exprience_df.groupby('MSISDN/Number').agg({'Total Avg RTT (ms)': 'sum', 'Total Avg Bearer TP (kbps)': 'sum', 'Total TCP Retrans. Vol (Bytes)': 'sum', 'Handset Type': [lambda x: x.mode()[0]]})  # ' '.join(x)
    tellco_exprience_df = pd.DataFrame(columns=["Total Avg RTT (ms)","Total Avg Bearer TP (kbps)","Total TCP Retrans. Vol (Bytes)","Handset Type"])
    tellco_exprience_df["Total Avg RTT (ms)"] = tellco_exprience_df1["Total Avg RTT (ms)"]['sum']
    tellco_exprience_df["Total Avg Bearer TP (kbps)"] = tellco_exprience_df1["Total Avg Bearer TP (kbps)"]['sum']
    tellco_exprience_df["Total TCP Retrans. Vol (Bytes)"] = tellco_exprience_df1["Total TCP Retrans. Vol (Bytes)"]['sum']
    tellco_exprience_df["Handset Type"] = tellco_exprience_df1["Handset Type"]['<lambda>']
    option = st.selectbox('Top 10 of the top, bottom and most frequent Datas Based on',('Total Avg RTT (ms)', 'Total Avg Bearer TP (kbps)', 'Total TCP Retrans. Vol (Bytes)'))
    data = tellco_exprience_df.sort_values(option, ascending=False)
    highest = data.head(10)[option]
    lowest = data.tail(10)[option]
    most = tellco_exprience_df[option].value_counts().head(10)
    st.header("Highest")
    highest = highest.reset_index('MSISDN/Number')
    fig = px.bar(highest, x='MSISDN/Number', y=option)
    fig.update_layout(xaxis_type='category')
    st.plotly_chart(fig)
    st.header("Lowest")
    lowest = lowest.reset_index('MSISDN/Number')
    fig = px.bar(lowest, x='MSISDN/Number', y=option)
    fig.update_layout(xaxis_type='category')
    st.plotly_chart(fig)
    st.header("Most")
    st.dataframe(most)
    st.title("User Clusters")
    st.write("")
    st.dataframe(user_experience_data.head(1000))
    st.write("")
    st.write("")
    st.markdown("***Users classified into 3 clusters based on their experience(i.e. average RTT, TCP retransmission', and throughput).***")
    fig = px.scatter(user_experience_data, x='Total TCP Retrans. Vol (Bytes)', y='Total Avg Bearer TP (kbps)',color='cluster', size='Total Avg RTT (ms)')
    st.plotly_chart(fig)

    

def user_satisfaction():
    st.title("User Satisfaction Analysis")
    st.write("")
    st.header("User engagement score table")
    eng_df = user_satisfaction_score_data[['MSISDN/Number','xDR Sessions', 'Dur. (ms)', 'Total Data Volume (Bytes)', 'engagement_score']]
    sat_score_df_agg = user_satisfaction_score_data.groupby('MSISDN/Number').agg({'Dur. (ms)': 'sum', 'Total Data Volume (Bytes)': 'sum','engagement_score':'sum','engagement_cluster':'sum','Total Avg RTT (ms)':'sum','Total Avg Bearer TP (kbps)':'sum','Total TCP Retrans. Vol (Bytes)':'sum','experience_score':'sum','experience_cluster':'sum','satisfaction_score':'sum',})
    st.write(eng_df.head(1000))
    st.write("")
    st.markdown("**Users classified into 6 clusters based on their engagement(i.e. number of xDR sessions, duration and total data used).**")
    fig = px.scatter(user_satisfaction_score_data, x='Total Data Volume (Bytes)', y='Dur. (ms)',color='engagement_cluster', size='xDR Sessions')
    st.plotly_chart(fig)
    st.write("")
    st.header("User experience score table")
    exp_df = user_satisfaction_score_data[['MSISDN/Number', 'Total Avg RTT (ms)','Total Avg Bearer TP (kbps)', 'Total TCP Retrans. Vol (Bytes)', 'experience_score']]
    st.write(exp_df.head(1000))
    st.write("")
    st.markdown("**Users classified into 3 clusters based on their experience(i.e. average RTT, TCP retransmission', and throughput).**")
    fig = px.scatter(user_satisfaction_score_data, x='Total TCP Retrans. Vol (Bytes)', y='Total Avg Bearer TP (kbps)',color='experience_cluster', size='Total Avg RTT (ms)')
    st.plotly_chart(fig)
    st.write("")
    st.header("User satisfaction score table")
    sat_df = user_satisfaction_score_data[['MSISDN/Number', 'engagement_score', 'experience_score', 'satisfaction_score']]
    st.write(sat_df.head(1000))
    st.write("")
    st.markdown("**Users classified into 2 clusters based on their satisfactio(i.e. engagement score and experience score).**")
    fig = px.scatter(user_satisfaction_score_data, x='engagement_score', y='experience_score',color='satisfaction_cluster', size='satisfaction_score')
    st.plotly_chart(fig)
    st.write("")
    st.header('Top 10 Numbers (Users) with highest')
    option = st.selectbox('Top 10 Numbers (Users) with highest',('Engagement Score', 'Experience Score', 'Satisfaction Score'))

    if option == 'Engagement Score':
        data = sat_score_df_agg.sort_values('engagement_score', ascending=False).head(10)
        name = 'engagement_score'
    elif option == 'Experience Score':
        data = sat_score_df_agg.sort_values('experience_score', ascending=False).head(10)
        name = 'experience_score'
    else:
        data = sat_score_df_agg.sort_values('satisfaction_score', ascending=False).head(10)
        name = 'satisfaction_score'
    data = data.reset_index('MSISDN/Number')
    st.write("")
    st.dataframe(data)


def model():
    st.title("Predict Satisfaction")
    st.write(regression_model(clean_data_df))


page_names_to_funcs = {
    
    "User Overview Analysis":user_overview,
    "User Engagement Analysis": user_engagement,
    "User Experience Analysis": user_experience,
    "User Satisfaction Analysis": user_satisfaction,
    "Predict Satisfaction":model,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()