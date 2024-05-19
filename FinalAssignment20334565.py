import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

weekdays = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

# Plot dataset
# Read CSV
pre_pand_df = pd.read_csv('PrePandemicdata.csv')
pand_df = pd.read_csv('Pandemicdata.csv')
post_pand_df = pd.read_csv('PostPandemicdata.csv')


# Calculate pandemic values
pand_start = pd.to_datetime("27-03-2020")
pand_end = pd.to_datetime("28-01-2022")
pre_pandemic_X = pd.to_datetime(pre_pand_df['Date'])
post_pandemic_X = pd.to_datetime(post_pand_df['Date'])
pandemic_X = pd.to_datetime(pand_df['Date'])
pre_pandemic_y = pre_pand_df['Bike Usage']
post_pandemic_y = post_pand_df['Bike Usage']
pandemic_y = pand_df['Bike Usage']

# Missing Data filler fill in missing values with yearly average
def fill_missing_data(df):
    usage = df.get("Bike Usage")
    total = usage.sum()
    dataset_avg = total/len(df)
    dates = pd.to_datetime(df['Date'])
    filled_dataset = pd.DataFrame()
    temp_dataset = pd.DataFrame()
    last_date = dates.iloc[0]
    last_start = 0
    for index in range(1,len(df)):
        current_date = dates.iloc[index]
        # If missing date
        if (current_date - last_date) != timedelta(days=1):
            filled_dataset = pd.concat([filled_dataset, df[last_start:index]], ignore_index=True)
            diff = current_date - last_date
            last_recorded_date = last_date
            while diff > timedelta(days=1):
                last_recorded_date = last_recorded_date + timedelta(days=1)
                is_weekend = last_recorded_date.dayofweek > 4
                if is_weekend:
                    data = {'Date': [last_recorded_date], 'Bike Usage': [3500], 'Day': [0]}
                else:
                    data = {'Date': [last_recorded_date], 'Bike Usage': [7500], 'Day': [0]}
                temp_dataset = pd.DataFrame.from_dict(data)
                filled_dataset = pd.concat([filled_dataset, temp_dataset], ignore_index=True)
                diff = diff - timedelta(days=1)
            last_start = index
        last_date = current_date
    filled_dataset = pd.concat([filled_dataset, df[last_start:index]], ignore_index=True)
    return filled_dataset, dataset_avg

pre_pand_df_filled, pre_pand_dataset_avg = fill_missing_data(pre_pand_df)
pand_df_filled, pand_dataset_avg = fill_missing_data(pand_df)



# UNCOMMENT WHEN FINISHED !!!
# Plot data
plt.rc('font', size=12)
plt.plot(pandemic_X, pandemic_y, color='red', label='Pandemic')
plt.plot(pre_pandemic_X,pre_pandemic_y, color='green', label='Pre-Pandemic')
plt.plot(post_pandemic_X,post_pandemic_y, color='blue', label='Post-Pandemic')
plt.xlabel("Date")
plt.ylabel("Number of Bikes used")
plt.title('Dublin bike usage 2019-2023')
plt.legend(loc='upper left')
plt.show()
plt.clf()


# Assess what bike usage might have been for the pandemic period if the pandemic had not happened
# split into training and testing data
# q = n step ahead prediction
# dd = number of samples per unit of time
# lag = number of features we want
# y = data points for time
# t = time in required units
# dt = difference in time

def test_preds(q,dd,lag,plot,y, t, dt, title, Kfold_ridge):
    #q−step ahead prediction
    stride=1
    XX=y[0:y.size-q-lag*dd:stride]
    # computes the time series feature values for each datapoint
    for i in range(1,lag):
        X=y[i*dd:y.size-q-(lag-i)*dd:stride]
        XX=np.column_stack((XX,X))
    yy=y[lag*dd+q::stride]; tt=t[lag*dd+q::stride]
    # reset index so data is alligned
    yy = yy.reset_index(drop=True)
    tt = tt.reset_index(drop=True)

    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)
    if plot:
        y_pred = model.predict(XX)
        plt.scatter(t, y, color='black', label= 'training data')
        plt.scatter(tt, y_pred, color='blue', label= 'predictions')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions"],loc='upper right')
        plt.title(title)
        plt.show()
        plt.clf()
    if Kfold_ridge:
        mean_error = [];
        std_error = []
        Ci_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        ###
        scaler = StandardScaler()
        XX = scaler.fit_transform(XX)
        yy = yy.to_numpy()
        yy = yy.reshape(-1, 1)
        yy = scaler.fit_transform(yy)
        ###
        for Ci in Ci_range:
            model = Ridge(alpha=1 / (2 * Ci))
            temp = []
            kf = KFold(n_splits=5)
            for train, test in kf.split(X):


                model.fit(XX[train], yy[train])
                ypred = model.predict(XX[test])
                from sklearn.metrics import mean_squared_error
                tmp = y[test]
                temp.append(mean_squared_error(yy[test], ypred))
            mean_error.append(np.array(temp).mean())
            std_error.append(np.array(temp).std())

        plt.errorbar(Ci_range, mean_error, yerr=std_error)
        plt.xscale('log')
        plt.xlabel('Ci');
        plt.ylabel('Mean square error')
        plt.title("5-fold Ridge log C values (0.00001-10000)")
        plt.show()
        plt.clf()

# pre pandemic
y = pre_pandemic_y
t = pre_pandemic_X
dt = 86400 #1 day interval (seconds)
# prediction using short−term trend
plot=True
# prediction using daily seasonality
d=math.floor(24*60*60/dt) # number of samples per day
# prediction using weekly seasonality

# Test PREDS FIRST
w=math.floor(7*24*60*60/dt) # number of samples per day
test_preds(q=1,dd=1,lag=3,plot=plot, y=y, t=t, dt=dt, title="1 step ahead predictions pre pandemic", Kfold_ridge=True)
test_preds(q=w,dd=w,lag=3,plot=plot, y=y, t=t, dt=dt, title="7 steps ahead predictions pre pandemic", Kfold_ridge=True)


# Predict values for the pandemic period
"""
def predict_future(q,dd,lag,plot,y, t, dt, title, Kfold_ridge):
    #q−step ahead prediction
    stride=1
    y_cpy = y   # REMOVE AT END
    XX=y[0:y.size-q-lag*dd:stride]
    # computes the time series feature values for each datapoint
    for i in range(1,lag):
        X=y[i*dd:y.size-q-(lag-i)*dd:stride]
        XX=np.column_stack((XX,X))
    yy=y[lag*dd+q::stride]; tt=t[lag*dd+q::stride]
    # reset index so data is alligned
    yy = yy.reset_index(drop=True)
    tt = tt.reset_index(drop=True)

    ### Fit to smaller scale
    scaler = StandardScaler()
    XX = scaler.fit_transform(XX)
    yy = yy.to_numpy()
    yy = yy.reshape(-1, 1)
    yy = scaler.fit_transform(yy)
    y = y.to_numpy()
    y = y.reshape(-1, 1)
    y = scaler.fit_transform(y)

    ###

    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
    model = Ridge(fit_intercept=False, alpha=1 / (2 * 0.001)).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)

    # for the length of the week
    next_week = pd.Series(index=np.arange(7))
    last_date = t.iloc[-1]
    for index in range(0,7):
        last_date = last_date + timedelta(days=1)
        next_week.iloc[index] = last_date

   # next_week = pd.to_datetime(next_week)

    #XXT = y[y.size - q - lag * dd:]
    #XXT = y[-((1 * dd) + q):]
    XXT = y[-dd:]
    for i in range(1, lag):
        X2 = y[-((i * dd) + q):-((i * dd))]
        XXT = np.column_stack((XXT, X2))

    y_pred_extra = model.predict(XXT)
    y_pred = model.predict(XX)

    # Convert back to correct scale
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_pred_extra = scaler.inverse_transform(y_pred_extra.reshape(-1, 1)).ravel()
    y = scaler.inverse_transform(y.reshape(-1, 1)).ravel()
    y =  pd.Series(y)
    y_pred = pd.Series(y_pred)
    #

    y_pred_extra = pd.Series(y_pred_extra)
    y_out = pd.concat([y,y_pred_extra], ignore_index=True)

    new_t = pd.concat([t,next_week], ignore_index=True)

    if plot:
        plt.scatter(t, y, color='black', label= 'training data')
        plt.scatter(tt, y_pred, color='blue', label= 'predictions')
        plt.scatter(next_week, y_pred_extra, color='red', label= 'next week predictions')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions","next weeks predictions"],loc='upper right')
        plt.title(title)
        plt.show()

        if Kfold_ridge:
            mean_error = [];
            std_error = []
            Ci_range = [0.000000000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
            ###
            scaler = StandardScaler()
            XX = scaler.fit_transform(XX)
            yy = yy.to_numpy()
            yy = yy.reshape(-1, 1)
            yy = scaler.fit_transform(yy)
            ###
            for Ci in Ci_range:
                model = Ridge(alpha=1 / (2 * Ci))
                temp = []
                kf = KFold(n_splits=5)
                for train, test in kf.split(X):
                    model.fit(XX[train], yy[train])
                    ypred = model.predict(XX[test])
                    from sklearn.metrics import mean_squared_error
                    tmp = y[test]
                    temp.append(mean_squared_error(yy[test], ypred))
                mean_error.append(np.array(temp).mean())
                std_error.append(np.array(temp).std())

                ypred_original_format = scaler.inverse_transform(ypred.reshape(-1, 1)).ravel()  # REMOVE THIS AT END!!!

            plt.errorbar(Ci_range, mean_error, yerr=std_error)
            plt.xscale('log')
            plt.xlabel('Ci');
            plt.ylabel('Mean square error')
            plt.title("5-fold Ridge log C values (0.00001-10000)")
            plt.show()
            plt.clf()

    return (y_out, new_t) #!!!
"""

def predict_future(q,dd,lag,plot,y, t, dt, title, Kfold_ridge):
    #q−step ahead prediction
    stride=1
    y_cpy = y   # REMOVE AT END
    XX=y[0:y.size-q-lag*dd:stride]
    # computes the time series feature values for each datapoint
    for i in range(1,lag):
        X=y[i*dd:y.size-q-(lag-i)*dd:stride]
        XX=np.column_stack((XX,X))
    yy=y[lag*dd+q::stride]; tt=t[lag*dd+q::stride]
    # reset index so data is alligned
    yy = yy.reset_index(drop=True)
    tt = tt.reset_index(drop=True)
    yy_compare = yy

    ### Fit to smaller scale
    scaler = StandardScaler()
    XX = scaler.fit_transform(XX)
    yy = yy.to_numpy()
    yy = yy.reshape(-1, 1)
    yy = scaler.fit_transform(yy)
    y = y.to_numpy()
    y = y.reshape(-1, 1)
    y = scaler.fit_transform(y)
    ###

    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
    model = Ridge(fit_intercept=False, alpha=1 / (2 * 0.001)).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)

    # for the length of the week
    next_week = pd.Series(index=np.arange(7))
    last_date = t.iloc[-1]
    for index in range(0,7):
        last_date = last_date + timedelta(days=1)
        next_week.iloc[index] = last_date

    XXT = y[-dd:]
    for i in range(1, lag):
        X2 = y[-((i * dd) + q):-((i * dd))]
        XXT = np.column_stack((XXT, X2))

    y_pred_extra = model.predict(XXT)
    y_pred = model.predict(XX)


    # Convert back to correct scale
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_pred_extra = scaler.inverse_transform(y_pred_extra.reshape(-1, 1)).ravel()
    y = scaler.inverse_transform(y.reshape(-1, 1)).ravel()
    y =  pd.Series(y)
    y_pred = pd.Series(y_pred)
    #

    y_pred_extra = pd.Series(y_pred_extra)
    y_out = pd.concat([y,y_pred_extra], ignore_index=True)

    new_t = pd.concat([t,next_week], ignore_index=True)

    if plot:

        # compare with baseline predictor (always predicts the average)
        arr = np.zeros(shape=(len(y_pred_extra), 1))
        for index in range(0, len(arr)):
            arr[index] = pre_pand_dataset_avg
        arr = arr.flatten()
        baseline_pred = pd.Series(arr)
        arr2 = np.zeros(shape=(len(y_pred), 1))
        for index in range(0, len(arr2)):
            arr2[index] = pre_pand_dataset_avg
        arr2 = arr2.flatten()
        baseline_pred2 = pd.Series(arr2)

        plt.scatter(t, y, color='black', label= 'training data')
        plt.scatter(tt, y_pred, color='blue', label= 'predictions')
        plt.scatter(next_week, y_pred_extra, color='red', label= 'next week predictions')
        plt.scatter(next_week, baseline_pred, color='green', label='baseline predictions')
        plt.scatter(tt, baseline_pred2, color='green')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions","next weeks predictions", "baseline predictions"],loc='upper right')
        plt.title("model predictions vs baseline predictions")
        plt.show()
        plt.clf()
        # MSE comparison of plotted data with a simple baseline
        predictions_error = mean_squared_error(yy_compare,y_pred)
        predictions_error = math.sqrt(predictions_error)
        print("Training data predictions RMSE = " + str(predictions_error))
        baseline_error = mean_squared_error(yy_compare, baseline_pred2)
        baseline_error = math.sqrt(baseline_error)

        print("Training data baseline RMSE = " + str(baseline_error))

    if Kfold_ridge:
        mean_error = [];
        std_error = []
        Ci_range = [0.000000000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        ###
        scaler = StandardScaler()
        XX = scaler.fit_transform(XX)
        #yy = yy.to_numpy()
        yy = yy.reshape(-1, 1)
        yy = scaler.fit_transform(yy)
        ###
        for Ci in Ci_range:
            model = Ridge(alpha=1 / (2 * Ci))
            temp = []
            kf = KFold(n_splits=5)
            for train, test in kf.split(X):
                model.fit(XX[train], yy[train])
                ypred = model.predict(XX[test])
                tmp = y[test]
                temp.append(mean_squared_error(yy[test], ypred))
            mean_error.append(np.array(temp).mean())
            std_error.append(np.array(temp).std())

        plt.errorbar(Ci_range, mean_error, yerr=std_error)
        plt.xscale('log')
        plt.xlabel('Ci');
        plt.ylabel('Mean square error')
        plt.title("5-fold Ridge log C values (0.00001-10000)")
        plt.show()
        plt.clf()


    return (y_out, new_t) #!!!

# Predict next 7 days, loop for pandemic period.
PANDEMIC_LENGTH = len(pand_df)
index = 0
plot = True
original_t = t
original_y = y

# Calculate pandemic values
pre_pandemic_X = pd.to_datetime(pre_pand_df_filled['Date'])
pre_pandemic_y = pre_pand_df_filled['Bike Usage']
y = pre_pandemic_y
t = pre_pandemic_X

# perform ridge regression on the model to determine C value
predict_future(q=w, dd=w, lag=52, plot=plot, y=y, t=t, dt=dt, title="", Kfold_ridge=True)

plot = False
while index < PANDEMIC_LENGTH:
    if index == 21:
        print("debug")
    (y, t) = predict_future(q=w, dd=w, lag=52, plot=plot, y=y, t=t, dt=dt, title="", Kfold_ridge=False)
    index = index + 7

# Plot data
plt.scatter(t, y, color='red', label='No pandemic predictions')
Pre_pandemic_time = t[t< datetime.strptime('27-03-2020', "%d-%m-%Y")]
plt.scatter(t[t< datetime.strptime('27-03-2020', "%d-%m-%Y")], y[:len(Pre_pandemic_time)], color='black')
plt.scatter(original_t, original_y, color='black', label='training data')

plt.xlabel("time (days)");
plt.ylabel("#bikes")
plt.legend(loc='upper right')
plt.title("Future predictions if pandemic had never happened")
plt.show()
plt.clf()

# Plot comparison of actual pandemic data, predicted data and baseline data
# pandemic data not filled as we do not predict off this data
plt.scatter(pandemic_X, pandemic_y, label='pandemic actual usage', color='orange')
plt.scatter(t, y, color='red', label='No pandemic predictions')
plt.scatter(t[t< datetime.strptime('27-03-2020', "%d-%m-%Y")], y[:len(Pre_pandemic_time)], color='black')
plt.scatter(original_t, original_y, color='black', label='training data')

plt.xlabel("time (days)");
plt.ylabel("#bikes")
plt.legend(loc='upper right')
plt.title("Future predictions if pandemic had never happened vs pandemic data")
plt.show()

