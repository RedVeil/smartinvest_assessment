import pandas as pd 
import numpy as np 
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import layout, row
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from data_preperation import prepare_data


def create_charts(y_predict, df2, cut_off, forward_lag, title):
    p1 = figure(plot_width=920, plot_height=402, x_axis_type="datetime")
    p1.line(df2[cut_off+forward_lag:]["Date"], y_predict[:-forward_lag], color="red", legend_label='Prediction')
    #p1.line(df2[cut_off+forward_lag:]["Date"], df2[cut_off+forward_lag:]["Close_change"], color="blue", legend_label='Real')
    p1.line(df2[cut_off+forward_lag:]["Date"], df2[cut_off+forward_lag:]["SMA_10"], color="darkgreen" ,legend_label='Sma_10')
    p1.title.text = f"{title}-Prediction"
    p1.legend.location = "top_left"
    p1.grid.grid_line_alpha = 0.1
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price'

    p2 = figure(plot_width=920, plot_height=402, x_axis_type="datetime")
    p2.line(df2[cut_off+forward_lag:]["Date"], y_predict[:-forward_lag]/df2[cut_off+forward_lag:]["Close_change"], color="red", legend_label='Correlation')
    p2.title.text = f"{title}-Correlation"
    p2.legend.location = "top_left"
    p2.grid.grid_line_alpha = 0.1
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Correlation'

    return p1,p2


def print_results(results, title):
    print(f"{title} Results")
    print(f"Mean Absolute Error: {results['mean_absolute_error']}")
    print(f"Train score: {results['train_score']}")
    print(f"Test score: {results['test_score']}")
    print("______________________________________________________________")

    #print("Feature coefficients")
    #for i in range(len(coefficients)):
    #    print(f"{x_train.columns.values[i]} : {coefficients[i]}")

def linear_regression(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train) 

    y_predict = model.predict(x_test)
    coefficients = model.coef_
    mean_abs_error = mean_absolute_error(y_test,y_predict)
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return_values = {
                    "predictions": y_predict, 
                    "coefficients": coefficients, 
                    'mean_absolute_error': mean_abs_error, 
                    'train_score': train_acc, 
                    'test_score': test_acc}
    return return_values

def ridge_regression(x_train, x_test, y_train, y_test,alpha=1,max_iter=1000):
    model = Ridge(alpha=alpha,max_iter=max_iter)
    model.fit(x_train, y_train) 

    y_predict = model.predict(x_test)
    coefficients = model.coef_
    mean_abs_error = mean_absolute_error(y_test,y_predict)
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return_values = {
                    "predictions": y_predict, 
                    "coefficients": coefficients, 
                    'mean_absolute_error': mean_abs_error, 
                    'train_score': train_acc, 
                    'test_score': test_acc}
    return return_values

def bayesian_ridge_regression(x_train, x_test, y_train, y_test):
    model = BayesianRidge()
    model.fit(x_train, y_train) 

    y_predict = model.predict(x_test)
    coefficients = model.coef_
    mean_abs_error = mean_absolute_error(y_test,y_predict)
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return_values = {
                    "predictions": y_predict, 
                    "coefficients": coefficients, 
                    'mean_absolute_error': mean_abs_error, 
                    'train_score': train_acc, 
                    'test_score': test_acc}
    return return_values

def nn_regression(x_train, x_test, y_train, y_test, hidden_layers=(100,), max_iter=200):
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=max_iter)
    model.fit(x_train, y_train) 

    y_predict = model.predict(x_test)
    mean_abs_error = mean_absolute_error(y_test,y_predict)
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return_values = {
                    "predictions": y_predict, 
                    'mean_absolute_error': mean_abs_error, 
                    'train_score': train_acc, 
                    'test_score': test_acc}
    return return_values





if __name__ == "__main__":
    forward_lag = 7
    x_train, x_test, y_train, y_test, df2, cut_off = prepare_data("MSFT", forward_lag=forward_lag)
    linear_regression_results = linear_regression(x_train, x_test, y_train, y_test)
    ridge_regression_results = ridge_regression(x_train, x_test, y_train, y_test)
    bayesian_ridge_regression_results = bayesian_ridge_regression(x_train, x_test, y_train, y_test)
    nn_regression_results = nn_regression(x_train, x_test, y_train, y_test, hidden_layers=(100,5))

    lin1, lin2 = create_charts(linear_regression_results["predictions"], df2, cut_off, forward_lag, "Linear Regression")
    rig1, rig2 = create_charts(ridge_regression_results["predictions"], df2, cut_off, forward_lag, "Ridge Regression")
    brig1, brig2 = create_charts(bayesian_ridge_regression_results["predictions"], df2, cut_off, forward_lag, "Bayesian Ridge Regression")
    nn1, nn2 = create_charts(nn_regression_results["predictions"], df2, cut_off, forward_lag, "NN Regression")

    window_size = 30
    window = np.ones(window_size)/float(window_size)
    chart_title = "Regression Models"
    output_file(f"{chart_title}2.html", title=chart_title)

    
    show(layout([[lin1, rig1],[brig1,nn1],[lin2, rig2],[brig2,nn2]]))
