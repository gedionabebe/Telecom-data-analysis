import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import mlflow
import logging
import pickle
import datetime


def randomforestregressor(x,y,n_estimators=35,max_depth=20,random_state=5):

    #Logger
    logging.basicConfig(filename='../log/log.log', filemode='a',encoding='utf-8', level=logging.DEBUG)

    #Initialize experiment and log parameters
    mlflow.set_experiment("Predicting User Satisfaction using Random Forest Regressor")
    mlflow.log_param('Model', 'Random Forest Regressor')
    mlflow.log_param('Number of estimators',n_estimators)
    mlflow.log_param('Max depth',max_depth)
    mlflow.log_param('Random state',random_state)

    #Test,train split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    #Build ML pipeline
    pipeline = Pipeline(steps = [('preprocessor', StandardScaler()),('model',RandomForestRegressor(n_estimators = n_estimators,max_depth=max_depth, random_state=random_state))])
    
    #Train Random Forest Regressor
    random_forest_model = pipeline.fit(X_train, y_train)
    print(random_forest_model.named_steps['preprocessor'])
    #Prediction and score
    y_pred= random_forest_model.predict(X_test)
    Score = random_forest_model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #Log metrics
    mlflow.log_metric('Score',Score)
    mlflow.log_metric('Mean Squared Error',mse)
    mlflow.log_metric('r2 score',r2)

    #Extract feature importance
    importance = random_forest_model.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame()
    fi_df['feature'] = X_train.columns.to_list()
    fi_df['feature_importances'] = importance

    #Save trained model
    random_forest_model_path = '../models/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.pkl'
    pickle.dump(random_forest_model, open(random_forest_model_path, 'wb'))

    return y_pred, fi_df, Score, mse, r2,random_forest_model.named_steps['preprocessor']