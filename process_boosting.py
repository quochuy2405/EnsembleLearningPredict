from csv import writer
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

 # list of column names
field_names = ['formatted_date', 'high', 'low',
            'open', 'close', 'volume', 'adjclose']
            
pre_day = 30

MA_1 = 7
MA_2 = 14
MA_3 = 21

cols_y_close = ['close']
cols_y_open = ['open']
cols_y_high = ['high']
cols_y_low = ['low']



cols_x_close = ['high','low','open','H-L', 'O-C', f'SMA_{MA_1}_close', f'SMA_{MA_2}_close', f'SMA_{MA_3}_close', f'SD_{MA_1}_close', f'SD_{MA_3}_close']
cols_x_open = ['high','low','close','H-L', 'O-C', f'SMA_{MA_1}_open', f'SMA_{MA_2}_open', f'SMA_{MA_3}_open', f'SD_{MA_1}_open', f'SD_{MA_3}_open']
cols_x_high = ['low','open','close','H-L', 'O-C', f'SMA_{MA_1}_high', f'SMA_{MA_2}_high', f'SMA_{MA_3}_high', f'SD_{MA_1}_high', f'SD_{MA_3}_high']
cols_x_low = ['high','open','close','H-L', 'O-C', f'SMA_{MA_1}_low', f'SMA_{MA_2}_low', f'SMA_{MA_3}_low', f'SD_{MA_1}_low', f'SD_{MA_3}_low']

def loadModel():
    model_close = load_model("./model/BOOSTING/GRU_boosting_close.h5")
    model_open = load_model("./model/BOOSTING/GRU_boosting_open.h5")
    model_high = load_model("./model/BOOSTING/GRU_boosting_high.h5")
    model_low = load_model("./model/BOOSTING/GRU_boosting_low.h5")
    return model_close,model_open ,model_high ,model_low


def loadModelError():
    model_error_close = pickle.load(open("./model/BOOSTING/booting_gradient_close.h5",'rb'))
    model_error_open = pickle.load(open("./model/BOOSTING/booting_gradient_open.h5",'rb'))
    model_error_high = pickle.load(open("./model/BOOSTING/booting_gradient_high.h5",'rb'))
    model_error_low = pickle.load(open("./model/BOOSTING/booting_gradient_low.h5",'rb'))
    return model_error_close,model_error_open ,model_error_high ,model_error_low

# Prepare variables
def prepareVariable(df,MA_1,MA_2,MA_3):
    df['H-L'] = df['high'] - df['low']
    df['O-C'] = df['open'] - df['close']

    # Open
    df[f'SMA_{MA_1}_open'] = df['open'].rolling(window=MA_1).mean()
    df[f'SMA_{MA_2}_open'] = df['open'].rolling(window=MA_2).mean()
    df[f'SMA_{MA_3}_open'] = df['open'].rolling(window=MA_3).mean()

    df[f'SD_{MA_1}_open'] = df['open'].rolling(window=MA_1).std()
    df[f'SD_{MA_3}_open'] = df['open'].rolling(window=MA_3).std()

    # Close
    df[f'SMA_{MA_1}_close'] = df['close'].rolling(window=MA_1).mean()
    df[f'SMA_{MA_2}_close'] = df['close'].rolling(window=MA_2).mean()
    df[f'SMA_{MA_3}_close'] = df['close'].rolling(window=MA_3).mean()

    df[f'SD_{MA_1}_close'] = df['close'].rolling(window=MA_1).std()
    df[f'SD_{MA_3}_close'] = df['close'].rolling(window=MA_3).std()

    # High
    df[f'SMA_{MA_1}_high'] = df['high'].rolling(window=MA_1).mean()
    df[f'SMA_{MA_2}_high'] = df['high'].rolling(window=MA_2).mean()
    df[f'SMA_{MA_3}_high'] = df['high'].rolling(window=MA_3).mean()


    df[f'SD_{MA_1}_high'] = df['high'].rolling(window=MA_1).std()
    df[f'SD_{MA_3}_high'] = df['high'].rolling(window=MA_3).std()

    # Low
    df[f'SMA_{MA_1}_low'] = df['low'].rolling(window=MA_1).mean()
    df[f'SMA_{MA_2}_low'] = df['low'].rolling(window=MA_2).mean()
    df[f'SMA_{MA_3}_low'] = df['low'].rolling(window=MA_3).mean()

    df[f'SD_{MA_1}_low'] = df['low'].rolling(window=MA_1).std()
    df[f'SD_{MA_3}_low'] = df['low'].rolling(window=MA_3).std()
    df.dropna(inplace=True)
    df.to_csv("./data/BOOSTING/btc_pred_process.csv", index=False)

# processing data
def processingData(df,cols_y_close,cols_y_open,cols_y_high,cols_y_low):
    scala_x = MinMaxScaler(feature_range=(0,1))
    scala_y = MinMaxScaler(feature_range=(0,1))

    scaled_data_x_close = scala_x.fit_transform(df[cols_x_close].values.reshape(-1, len(cols_x_close)))
    scaled_data_x_open = scala_x.fit_transform(df[cols_x_open].values.reshape(-1, len(cols_x_open)))
    scaled_data_x_high = scala_x.fit_transform(df[cols_x_high].values.reshape(-1, len(cols_x_high)))
    scaled_data_x_low = scala_x.fit_transform(df[cols_x_low].values.reshape(-1, len(cols_x_low)))

    scaled_data_y_close = scala_y.fit_transform(df[cols_y_close].values.reshape(-1, len(cols_y_close)))
    scaled_data_y_open = scala_y.fit_transform(df[cols_y_open].values.reshape(-1, len(cols_y_open)))
    scaled_data_y_high = scala_y.fit_transform(df[cols_y_high].values.reshape(-1, len(cols_y_high)))
    scaled_data_y_low = scala_y.fit_transform(df[cols_y_low].values.reshape(-1, len(cols_y_low)))
    return scaled_data_x_close,scaled_data_x_open,scaled_data_x_high,scaled_data_x_low,scaled_data_y_close,scaled_data_y_open,scaled_data_y_high,scaled_data_y_low,scala_x,scala_y

def predict(df, cols_x, scala_x,scala_y, model,error_model):
    pre_day = 30
    scaled_data_x = scala_x.fit_transform(df[len(df)-pre_day:][cols_x].values.reshape(-1, len(cols_x)))  
    x_last=np.array([scaled_data_x])
    
    predict_price = model.predict(x_last)
    predict_price = scala_y.inverse_transform(predict_price)
    predict_error= error_model.predict(np.array(predict_price).reshape(-1,1))[0]
    return predict_error+predict_price[0][0]
def getPrediction(df, number_day, name, prediction): 
    df_3 = df[-number_day:][name].append(pd.Series([prediction]))
    mean = df_3.mean()
    if (number_day == 14):
        std = 0
    else:
        std = df_3.std()
    return mean, std
def getGeneratedColumns(df,close_pred, open_pred, high_pred, low_pred, number_day):
    mean_close, std_close = getPrediction(df, number_day, "close", close_pred)
    mean_open, std_open = getPrediction(df, number_day, "open", open_pred)
    mean_high, std_high = getPrediction(df, number_day, "high", high_pred)
    mean_low, std_low = getPrediction(df, number_day, "low", low_pred)
    return mean_close, std_close, mean_open, std_open, mean_high, std_high, mean_low, std_low
if __name__ == '__main__':
   
    # load dataframe
    df = pd.read_csv("./data/btc.csv")
    # load model
    model_close,model_open ,model_high ,model_low=loadModel()
    model_error_close,model_error_open ,model_error_high ,model_error_low= loadModelError()
    prepareVariable(df,MA_1,MA_2,MA_3)
    scaled_data_x_close,scaled_data_x_open,scaled_data_x_high,scaled_data_x_low,scaled_data_y_close,scaled_data_y_open,scaled_data_y_high,scaled_data_y_low,scala_x,scala_y=processingData(df,cols_y_close,cols_y_open,cols_y_high,cols_y_low)
    n=365
    df3 = pd.read_csv("./data/BOOSTING/btc_pred_process.csv")
    for i in range(n):
            pred = []
            low_pred = predict(df3, cols_x_low, scala_x,scala_y, model_low,model_error_low) 
            close_pred = predict(df3, cols_x_close, scala_x,scala_y, model_close,model_error_close) 
            open_pred = predict(df3, cols_x_open, scala_x,scala_y, model_open,model_error_open) 
            high_pred = predict(df3, cols_x_high, scala_x,scala_y, model_high,model_error_high) 

            high_low = high_pred - low_pred
            open_close = open_pred - close_pred

            mean_close_7, std_close_7, mean_open_7, std_open_7, mean_high_7, std_high_7, mean_low_7, std_low_7 = getGeneratedColumns(df3,close_pred, open_pred, high_pred, low_pred, 7)

            mean_close_14, a, mean_open_14, b, mean_high_14, c, mean_low_14, d = getGeneratedColumns(df3,close_pred, open_pred, high_pred, low_pred, 14)

            mean_close_21, std_close_21, mean_open_21, std_open_21, mean_high_21, std_high_21, mean_low_21, std_low_21 = getGeneratedColumns(df3,close_pred, open_pred, high_pred, low_pred, 21)
            # next day
            next_day = (dt.date.today() + dt.timedelta(days=i+1)).strftime("%Y-%m-%d")
            pred = [next_day, high_pred, low_pred,
                    open_pred, close_pred , 0, 0, high_low, open_close,
                    mean_open_7, mean_open_14, mean_open_21, std_open_7, std_open_21,
                    mean_close_7, mean_close_14, mean_close_21, std_close_7, std_close_21,
                    mean_high_7, mean_high_14, mean_high_21, std_high_7, std_high_21,
                    mean_low_7, mean_low_14, mean_low_21, std_low_7, std_low_21]
            df3.loc[len(df3)] = pred
    df3.set_index('formatted_date')
    df3.to_csv('./predict/BOOSTING/final_pred_boosting_gru_gradient.csv')
        