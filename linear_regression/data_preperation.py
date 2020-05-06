import pandas as pd 
import numpy as np
from sklearn.utils import shuffle

#  .shift(1) takes row with lower index
#  .shift(-1) takes row with higher index


def clean_data(df):
    df["Intraday_change"] = df["Open"]/df["Close"] 
    df["Close"] = df["Adjusted_close"]
    df = df.drop("Adjusted_close",axis=1)
    return df


def create_change_ranges(df):
    df["Maximum_change"] = df["Low"]/df["High"]
    df["Open_change"] = df["Open"].shift(1)/df["Open"]
    df["Close_change"] = df["Close"].shift(1)/df["Close"]
    return df

def create_sma(df):
    for sma_period in [5,10,20,50,100,200]:
        indicator_name = "SMA_%d" % (sma_period)
        df[indicator_name] = df['Close'].rolling(sma_period).mean()
    return df

def create_boilinger(df):
    for period in [10,20]:
        for deviation in [1,2]:
            up_name = f'BollingerBand_Up_{period}_{deviation}'
            down_name = f'BollingerBand_Down_{period}_{deviation}'
            df[up_name] = df['Close'].rolling(period).mean() + deviation*df['Close'].rolling(period).std()
            df[down_name] = df['Close'].rolling(period).mean() - deviation*df['Close'].rolling(period).std()
    return df

def create_donchian_channel(df):
    for channel_period in [5,10,20,50,100,200]:
        up_name = "Donchian_Channel_Up_%d" % (channel_period)
        down_name = "Donchian_Channel_Down_%d" % (channel_period)  
        df[up_name] = df['High'].rolling(channel_period).max()
        df[down_name] = df['Low'].rolling(channel_period).min()
    return df

def normalize(df):
    blub = ["Volume","Intraday_change","Maximum_change","Open_change","Close_change"]
    for column in df.columns.values:
        if column not in blub:
            print(column)
            df[column] = (df[column] - df["Donchian_Channel_Down_200"])/(df["Donchian_Channel_Up_200"]-df["Donchian_Channel_Down_200"])
        else:
            if column != "Volume":
                df[column] = (df[column] - df[column].min() ) / (df[column].min()-df[column].max())
    return df


def create_target(df, forward_lag=5):
    df['Target'] = df['Close_change'].shift(-forward_lag)
    df = df.dropna()
    df2 = df.copy()
    df = df.drop(['Close_change',"Date"],axis=1)  #
    return df, df2

def create_indicator(df):
    df = create_sma(df)
    df = create_boilinger(df)
    df = create_donchian_channel(df)
    return df


def prepare_data(stock, forward_lag=7, set_shuffle=False, set_normalize=False):
    df = pd.read_csv(f"../data/{stock}_US.csv",parse_dates=["Date"])
    #df2 = pd.read_csv("./AAPL_US.csv")
    #df = df.append(df2)
    df = clean_data(df)
    df = create_change_ranges(df)
    df = create_indicator(df)
    df,df2 = create_target(df,forward_lag)
    if set_normalize:
        df = normalize(df)
    cut_off = int(len(df)-(len(df)*0.2))
    train = df[:cut_off]
    test = df[cut_off:]
    if set_shuffle:
        train = shuffle(train).reset_index(drop=True)
    x_train = train.drop("Target",axis=1)
    y_train = train['Target']
    x_test = test.drop("Target",axis=1)
    y_test = test['Target']
    return x_train, x_test, y_train, y_test, df2, cut_off
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
