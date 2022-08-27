import numpy as np
import pandas as pd
import streamlit as st
from data_extraction import Data_extract
from data_analysis import plot_hist, plot_heatmap,handset_manufacturer,data_info
from satisfaction_analysis import engagement_score, experience_score, satisfaction_score
from user_engagement_analysis import total_app_traffic
from experience_analytics import experience_metrics, metrics_per_handset
from model import regression_model

data= Data_extract()

def user_overview():
    st.title("User Overview Analysis")
    st.write(handset_manufacturer(data))
    st.write(data_info(data))

def user_engagement():
    st.title("User Engagement Analysis")
    st.write(engagement_score(data))
    st.write(total_app_traffic(data,'Gaming DL (Bytes)','Gaming UL (Bytes)','Gaming Total (Bytes)'))

def user_experience():
    st.title("User Experience Analysis")
    st.write(experience_metrics(data))
    

def user_satisfaction():
    st.title("User Satisfaction Analysis")
    st.write(engagement_score(data))
    st.write(experience_score(data))
    x=experience_score(data)
    y=engagement_score(data)
    st.write(satisfaction_score(x,y))

def model():
    st.title("Predict Satisfaction")
    st.write(regression_model(data))


page_names_to_funcs = {
    
    "User Overview Analysis":user_overview,
    "User Engagement Analysis": user_engagement,
    "User Experience Analysis": user_experience,
    "User Satisfaction Analysis": user_satisfaction,
    "Predict Satisfaction":model,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()