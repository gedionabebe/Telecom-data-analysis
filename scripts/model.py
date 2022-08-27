import pandas as pd 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from satisfaction_analysis import satisfaction_score, engagement_score, experience_score
from data_cleaning_transformation import standard_scaler


def regression_model(data):
    
    eng = engagement_score(data)
    exp = experience_score(data)
    x= pd.DataFrame(list(zip(eng['session_freq'],eng['session_dur'],eng['session_data'],exp['avg_tcp'],exp['avg_rtt'],exp['hand_sets'],exp['avg_throughput'])),columns=['session_freq','session_dur','session_data','avg_tcp','avg_rtt','hand_sets','avg_throughput'])
    y= satisfaction_score(exp,eng)['satisfaction_score'].to_list()
    X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=1, test_size=0.3)
    X_trainscaled = standard_scaler(X_train)
    X_testscaled = standard_scaler(X_test)
    reg_model = MLPRegressor(hidden_layer_sizes=(20,20,20),activation="relu" ,random_state=1, max_iter=100).fit(X_trainscaled, y_train)
    y_prediction = reg_model.predict(X_testscaled)
    

    return y_prediction
