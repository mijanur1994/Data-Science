import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas_datareader as web
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as pdr
import datetime
from sklearn.preprocessing import StandardScaler

ticker = "TCS.NS"
df = pdr.get_data_yahoo(ticker,datetime.date.today()-datetime.timedelta(380),datetime.date.today())

df.head()

dataset=df['01-04-2019':'31-03-2020']
dataset.head()

dataset.reset_index(level=0,inplace=True)


dataset.head()



dataset.to_csv('dataset1',index=False,header=True)


bk= pd.read_csv('dataset1')
bk.head()

x= dataset.iloc[:,0:1]
y=dataset.iloc[:,3:4]

############################
x1 = dataset.iloc[:, 2:3]
y1 = dataset.iloc[:, 0:1]
############################
''' Tui j eerror ta send korechilis seta ei data diye eivabe handle kora jbe
x1 variable sudhu low price ta r y1 variable Date ta store ache...
'''


dates= []
prices=[]

dataset.shape

dataset.info()

scaler= StandardScaler()
features_standardized=scaler.fit_transform(y)

features_standardized

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=10)


model = SVR()
model.fit(X_train,y_train)


from sklearn.model_selection import GridSearchCV
gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 2,5,10, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=10, n_jobs=-1)

grid_result = gsc.fit(x, y)
best_params = grid_result.best_params_




