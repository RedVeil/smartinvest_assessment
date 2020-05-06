import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)
import random, timeit
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

def load_data():
    #df = pd.read_csv("./data/modified_MSFT.csv")
    #price = df["SMA_5"]
    price = np.sin(np.arange(400)/30.0) 
    price = [i+2 for i in price]
    diff = np.diff(price)
    diff = np.insert(diff, 0, 0)
    data = np.column_stack((price, diff))
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    cut_off = int(len(data)-(len(data)*0.2))
    train = data[:cut_off]
    test = data[cut_off:]
    return train, test, price


def get_reward(df, i, gamma, action):
    #--- Maybe reward more than 0 for not beeing in bad trades
    if i == 0:
        reward = 1*gamma
    else:   
        reward = ((df["price"][i-1] - df["price"][i-1])*100)*action      
    reward *= gamma
    return reward


def run_simulation(data, df, model, epsilon):
    for i in range(len(data)):
        if i == len(data)+2: #if i == len(data)-1:
            pass
            #get rewards of the model at the last datapoint
        else:
            state = data[i:i+1, :] #---!!! ask about shape
            qval = model.predict(state, batch_size=1)
            if (random.random() < epsilon):
                action = np.random.randint(0,2)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            df["position"][i] = action #df["position"][i] to make it realisticly
            reward = get_reward(df, i, gamma, action)
            qval[0][action] = reward
            model.fit(state, qval, batch_size=1, epochs=1, verbose=0)
    return model, df

def plot_model(df, epoch):
    plt.figure()
    df["strategy_pct"] = df["price"].pct_change(1) * df["position"]
    df["strategy"] = (df["strategy_pct"] + 1).cumprod()
    df["buy_hold"] = (df["price"].pct_change(1) + 1).cumprod()
    df[["strategy","buy_hold"]].plot()
    plt.savefig(f'./plt/iteration5/epoch{epoch}.png')
    plt.close()

'''
model = Sequential()
model.add(Dense(64,
               input_shape=(2,),
               #return_sequences=True,
               stateful=False))
model.add(Dropout(0.5))
model.add(Dense(64,
               input_shape=(2,),
               #return_sequences=False,
               stateful=False))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
rms = RMSprop()
adam = Adam()
model.compile(loss='mse', optimizer=adam)
'''

model = Sequential()

model.add(Dense(2, kernel_initializer='lecun_uniform', input_shape=(2,)))
model.add(Activation('relu'))

model.add(Dense(2, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(2, kernel_initializer='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

epochs = 51
gamma = 0.9 #a high gamma makes a long term reward more valuable
epsilon = 1
learning_progress = []


# split data for 20 epochs?
# 1. choose random action 1/0 for first action
# 2. move to second state
# 3. get reward for past action with todays state
# 4. choose action with todays state for tomorrow
# 5. repeat

start_time = timeit.default_timer()

train, test, price = load_data()
data = {"price":price, "position":np.zeros(len(price))}
df = pd.DataFrame(data=data)
for i in range(epochs):
    print(f"epoch{i}")
    if i == epochs -1: #test model in last run
        print("!!!test!!!")
        model,df  = run_simulation(test, df, model, epsilon)
        plot_model(df, i)
    else:
        model,df  = run_simulation(train, df, model, epsilon)
        plot_model(df, i)
    if epsilon > 0.05:
        epsilon -= (1.0/epochs)

elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))