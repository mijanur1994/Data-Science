import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser 

# import dataset
fineTech_appData = pd.read_csv("FineTech_appData.csv")

# Getting Shape
fineTech_appData.shape

# Showing first and last 10 rows from the dataset
fineTech_appData.head(10)
fineTech_appData.tail(10)

# The 6th number column’s (screen_list) full information not
# visible, so for that we used below python code snippet.
# We print only 5 rows from index 1 to 5 from the screen_list.
for i in [1,2,3,4,5]:
    print(fineTech_appData.loc[i,'screen_list'],'\n')
    
''''
Know about dataset:
As you can see in fineTech_appData DataFrame, there are 50,000 users data
with 12 different features. Let’s know each and every feature in brief.

1. user: Unique ID for each user.

2. first_open: Date (yy-mm-dd) and time (Hour:Minute:Seconds:Milliseconds)
of login on app first time.

3. dayofweek: On which day user logon.
0: Sunday
1: Monday
2: Tuesday
3: Wednesday
4: Thursday
5: Friday
6: Saturday

4. Hour: Time of a day in 24-hour format customer logon. It is correlated
with dayofweek column.

5. age: The age of the registered user.

6. screen_list: The name of multiple screens seen by customers, which are
separated by a comma.

7. numscreens: The total number of screens seen by customers.

8. minigame: Tha app contains small games related to finance. If the customer
played mini-game then 1 otherwise 0.

9. used_premium_feature: If the customer used the premium feature of the app
then 1 otherwise 0.

10. enrolled: If the user bought a premium feature app then 1 otherwise 0.

11. enrolled_date: On the date (yy-mm-dd) and time
(Hour:Minute:Seconds:Milliseconds) the user bought a premium features app.

12. liked: The each screen of the app has a like button if the customer likes
it then 1 otherwise 0.'''

# Find the null value in DataFrame using DataFrame.isnull() method and take summation by sum() method.
fineTech_appData.isnull().sum()

'''All columns contain 0 null value except enrolled_date. The enrolled_date
column has total 18926 null values.'''


#Take brief information about the dataset using DataFrame.info() method.
fineTech_appData.info()
'''We can see in the output provided by DataFrame.info() method, there are 50,
000 entries (rows) from 0 to 49999 and a total of 12 columns.
All columns have 50,000 non-null values except enrolled_date. It has 31,074
non-null. There is a total of 8 columns that contain integer 64 bit (int64) values and the remaining 4 are object type.
The size of fineTech_appData DataFrame is 4.6 MB.'''

# The distribution of numerical variables
fineTech_appData.describe()
'''To know how the numeric variable distributed, we used DataFrame.describe()
method. It gives total number count, mean value, std (standard deviation),
min and max value, and values are below 25%, 50%, 75% of each column.

From the output, we can know more about the dataset. The mean age of the
customer is 31.72. Only 10.7% of customers played minigame and 17.2% customer
used premium features of the app, likes 16.5 %. The 62.1% customer enrolled
in the premium app.'''

# If you observe the description of ‘dayofweek’ column then you can not
# get proper information. To solve this issue we print unique values
# of each column and its length.
# Get the unique value of each columns and it's length
features = fineTech_appData.columns
for i in features:
    print("""Unique value of {}\n{}\nlen is {} \n........................\n
          """.format(i, fineTech_appData[i].unique(), len(fineTech_appData[i].unique())))
'''In the above output, we got information about the ‘dayofweek’ and ‘hour’
columns. The customer registers the app each day of the week and 24 hours.'''

# The ‘hour’ column contains object data type, so we converted into integer data type format.
#  hour data convert string to int
fineTech_appData['hour'] = fineTech_appData.hour.str.slice(1,3).astype(int) 
# get data type of each columns
fineTech_appData.dtypes

# Drop object dtype columns
fineTech_appData2 = fineTech_appData.drop(['user', 'first_open', 'screen_list', 'enrolled_date'], axis = 1)


############################## Data visualization ###########################

# Heatmap uses to find the correlation between each and every features using the correlation matrix.
# Heatmap
plt.figure(figsize=(16,9)) # heatmap size in ratio 16:9
sns.heatmap(fineTech_appData2.corr(), annot = True, cmap ='Greens_r') # show heatmap
plt.title("Heatmap using correlation matrix of fineTech_appData2", fontsize = 25) # title of heatmap

'''In the fineTech_appData2 dataset, there is no strong correlation
between any features. There is little correlation between ‘numscreens’
and ‘enrolled’.
It means that those customers saw more screen they are taking premium app.
There is a slight correlation between ‘minigame’ with ‘anrolled’ and
‘used_premium_feature’. The slightly negative correlation between ‘age’
with ‘enrolled’ and ‘numscreens’. It means that older customers do not
use the premium app and they don’t see multiple screens.'''

# Pair plot of fineTech_appData2
# The pair plot helps to visualize the distribution of data and scatter plot.
sns.pairplot(fineTech_appData2, hue  = 'enrolled')
'''In pair plot we can see, the maximum features have two values like 0 and 1
and orange dots show the enrolled customer’s features. So we visualize the
counterplot of enrolled data.'''

# Countplot of enrolled(Here you can see the exact value of enrolled & not enrolled customers.)
sns.countplot(fineTech_appData.enrolled)

# value enrolled and not enrolled customers
print("Not enrolled user = ", (fineTech_appData.enrolled < 1).sum(), "out of 50000")
print("Enrolled user = ",50000-(fineTech_appData.enrolled < 1).sum(),  "out of 50000")

# Histogram of each feature of fineTech_appData2

# plot histogram  
plt.figure(figsize = (16,9)) # figure size in ratio 16:9
features = fineTech_appData2.columns # list of columns name
for i,j in enumerate(features): 
    plt.subplot(3,3,i+1) # create subplot for histogram
    plt.title("Histogram of {}".format(j), fontsize = 15) # title of histogram
     
    bins = len(fineTech_appData2[j].unique()) # bins for histogram
    plt.hist(fineTech_appData2[j], bins = bins, rwidth = 0.8, edgecolor = "r", linewidth = 2, ) # plot histogram
     
plt.subplots_adjust(hspace=0.5) # space between horixontal axes (subplots)

'''In the above histogram, we can see minigame, used_primium_feature, enrolled,
and like they have only two values and how they distributed.
The histogram of ‘dayofweek’ shows, on Tuesday and Wednesday slightly fewer customer
registered the app.
The histogram of ‘hour’ shows the less customer register on the app around 10 AM.
The ‘age’ histogram shows, the maximum customers are younger.
The ‘numsreens’ histogram shows the few customers saw more than 40 screens.'''

# Correlation barplot with ‘enrolled’ feature
# show corelation barplot  
sns.set() # set background dark grid
plt.figure(figsize = (14,5))
plt.title("Correlation all features with 'enrolled' ", fontsize = 20)
fineTech_appData3 = fineTech_appData2.drop(['enrolled'], axis = 1) # drop 'enrolled' feature
ax =sns.barplot(fineTech_appData3.columns,fineTech_appData3.corrwith(fineTech_appData2.enrolled)) # plot barplot 
ax.tick_params(labelsize=15, labelrotation = 20, color ="k") # decorate x &amp; y ticks font

'''We saw the heatmap correlation matrix but this was not showing correlation
clearly but you can easily understand which feature is how much correlated with
‘enrolled’ feature using the above barplot.'''

# Now, we are parsing ‘first_open’ and ‘enrolled_date’ object data in data and time format.
# parsinf object data into data time format 
fineTech_appData['first_open'] =[parser.parse(i) for i in fineTech_appData['first_open']] 
fineTech_appData['enrolled_date'] =[parser.parse(i) if isinstance(i, str) else i for i in fineTech_appData['enrolled_date']] 
fineTech_appData.dtypes

# Showing the distribution of time taken to enrolled the app.
fineTech_appData['time_to_enrolled']  = (fineTech_appData.enrolled_date - fineTech_appData.first_open).astype('timedelta64[h]')
plt.hist(fineTech_appData['time_to_enrolled'].dropna())

# Let’s try to show the distribution in range 0 to 100 hours
# Plot histogram
plt.hist(fineTech_appData['time_to_enrolled'].dropna(), range = (0,100)) 


################################################################################
############################# Feature selection ################################
################################################################################
# We are considering those customers have enrolled after 48 hours as 0.
# Those customers have enrolled after 48 hours set as 0
fineTech_appData.loc[fineTech_appData.time_to_enrolled > 48, 'enrolled'] = 0
'''Drop some ‘time_to_enrolled’, ‘enrolled_date’, ‘first_open’ feature they are
not strongly correlated to the result.'''

fineTech_appData.drop(columns = ['time_to_enrolled', 'enrolled_date', 'first_open'], inplace=True)

# Read another CSV file that contains the top screens name
# read csv file and convert it into numpy array
fineTech_app_screen_Data = pd.read_csv("Dataset/FineTech appData/top_screens.csv").top_screens.values 
fineTech_app_screen_Data

#Add ‘,’ at the end of each string of ‘screen_list’ for further operation.
fineTech_appData['screen_list'] = fineTech_appData.screen_list.astype(str) + ','
'''The ‘Screen_list’ contains string values but we can’t use it directly.
So to solve this problem we are taking each screen name from
‘fineTech_app_screen_Data’ and append as a column by the same name to
‘fineTech_appData’. Then check this screen name is available in ‘screen_list’
if it is available then add value 1 else 0 in the appended column.'''

# string into to number 
for screen_name in fineTech_app_screen_Data:
    fineTech_appData[screen_name] = fineTech_appData.screen_list.str.contains(screen_name).astype(int)
    fineTech_appData['screen_list'] = fineTech_appData.screen_list.str.replace(screen_name+",", "")

# get shape
fineTech_appData.shape

# head of DataFrame
fineTech_appData.head(6)

# remain screen in 'screen_list'
fineTech_appData.loc[0,'screen_list']

# Droping ‘screen_list’ column.
fineTech_appData.drop(columns = ['screen_list'], inplace=True)

# TOTAL COLUMNS
fineTech_appData.columns

# take sum of all saving screen in one place
saving_screens = ['Saving1',
                  'Saving2',
                  'Saving2Amount',
                  'Saving4',
                  'Saving5',
                  'Saving6',
                  'Saving7',
                  'Saving8',
                  'Saving9',
                  'Saving10',
                 ]
fineTech_appData['saving_screens_count'] = fineTech_appData[saving_screens].sum(axis = 1)
fineTech_appData.drop(columns = saving_screens, inplace = True)

# Similarly for credit, CC1 and loan screens.

credit_screens = ['Credit1',
                  'Credit2',
                  'Credit3',
                  'Credit3Container',
                  'Credit3Dashboard',
                 ]
fineTech_appData['credit_screens_count'] = fineTech_appData[credit_screens].sum(axis = 1)
fineTech_appData.drop(columns = credit_screens, axis = 1, inplace = True)
1
2
3
4
5
6
cc_screens = ['CC1',
              'CC1Category',
              'CC3',
             ]

fineTech_appData['cc_screens_count'] = fineTech_appData[cc_screens].sum(axis = 1)
fineTech_appData.drop(columns = cc_screens, inplace = True)
1
2
3
4
5
6
7
loan_screens = ['Loan',
                'Loan2',
                'Loan3',
                'Loan4',
               ]

fineTech_appData['loan_screens_count'] = fineTech_appData[loan_screens].sum(axis = 1)
fineTech_appData.drop(columns = loan_screens, inplace = True)

# Shape of the dataframe
fineTech_appData.shape
fineTech_appData.info() #Information

# Numerical distribution of fineTech_appData
fineTech_appData.describe()

# Heat Map with the correlation matrix
# Heatmap with correlation matrix of new fineTech_appData 
plt.figure(figsize = (25,16)) 
sns.heatmap(fineTech_appData.corr(), annot = True, linewidth =2)

################################################################################
################################ Data Preprocessing ############################
################################################################################

# Split the dataset into train and test
clean_fineTech_appData = fineTech_appData
target = fineTech_appData['enrolled']
fineTech_appData.drop(columns = 'enrolled', inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(fineTech_appData, target, test_size = 0.2, random_state = 0)

# take User ID in another variable 
train_userID = X_train['user']
X_train.drop(columns= 'user', inplace =True)
test_userID = X_test['user']
X_test.drop(columns= 'user', inplace =True)

print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of train_userID = ', train_userID.shape)
print('Shape of test_userID = ', test_userID.shape)

###############################################################################
##################################### Feature Scaling #########################
###############################################################################
# Multiple features in the different units so for the best accuracy need to convert all features in a single unit
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Here target variable is categorical type 0 and 1, so we have to use supervised classification algorithms.
# we will build the best categorical type 0 and 1, so we have to use supervised classification algorithms.
# Even we canfind the best ML model. So let’s try.

# impoer required packages
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


##################### Decission Tree Classifier ################################
rom sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_score(y_test, y_pred_dt)

# train with Standard Scaling dataset
dt_model2 = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_model2.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


############################ K- Nearest Neighbor Classifier #######################
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
 
accuracy_score(y_test, y_pred_knn)

# train with Standert Scaling dataset
knn_model2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_knn_sc)


########################### Random Forest Classifiers ########################
# train with Standert Scaling dataset
knn_model2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_knn_sc)

# train with Standert Scaling dataset
rf_model2 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_rf_sc)


############################# Logistic regression #############################
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state = 0, penalty = 'l1')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
 
accuracy_score(y_test, y_pred_lr)

# train with Standert Scaling dataset
lr_model2 = LogisticRegression(random_state = 0, penalty = 'l1')
lr_model2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_lr_sc)


###################### Support Vector Classifiers #############################
# Support Vector Machine
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
 
accuracy_score(y_test, y_pred_svc)

# train with Standert Scaling dataset
svc_model2 = SVC()
svc_model2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_svc_sc)


######################## XG - Boost Classifier ################################
# XGBoost Classifier
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred_xgb)

# train with Standert Scaling dataset
xgb_model2 = XGBClassifier()
xgb_model2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_xgb_sc)


# XGB classifier with parameter tuning
xgb_model_pt1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 
xgb_model_pt1.fit(X_train, y_train)
y_pred_xgb_pt1 = xgb_model_pt1.predict(X_test)
 
accuracy_score(y_test, y_pred_xgb_pt1)

# XGB classifier with parameter tuning
# train with Stander Scaling dataset
xgb_model_pt2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 
xgb_model_pt2.fit(X_train_sc, y_train)
y_pred_xgb_sc_pt2 = xgb_model_pt2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_xgb_sc_pt2)



# Confusion Matrix
cm_xgb_pt2 = confusion_matrix(y_test, y_pred_xgb_sc_pt2)
sns.heatmap(cm_xgb_pt2, annot = True, fmt = 'g')
plt.title("Confussion Matrix", fontsize = 20) 

# The model is giving type II error higher than type I.
# Clasification Report
cr_xgb_pt2 = classification_report(y_test, y_pred_xgb_sc_pt2)
 
print("Classification report >>> \n", cr_xgb_pt2)

# Cross-validation of the ML model
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_model_pt2, X = X_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())

# Mapping predicted output to the target
final_result = pd.concat([test_userID, y_test], axis = 1)
final_result['predicted result'] = y_pred_xgb_sc_pt2
 
print(final_result)

# Save the Machine Learning model
'''After completion of the Machine Learning project or building the ML model
need to deploy in an application. To deploy the ML model need to save it first.
To save the Machine Learning project we can use the pickle or joblib package.'''

# Save the ML model with Pickle
## Pickle
import pickle
 
# save model
pickle.dump(xgb_model_pt2, open('FineTech_app_ML_model.pickle', 'wb'))
 
# load model
ml_model_pl = pickle.load(open('FineTech_app_ML_model.pickle', 'rb'))
# predict the output
y_pred_pl = ml_model.predict(X_test_sc)
# confusion matrix
cm_pl = confusion_matrix(y_test, y_pred)
print('Confussion matrix = \n', cm_pl)
# show the accuracy
print("Accuracy of model = ",accuracy_score(y_test, y_pred_pl))

# Save the Ml model with Joblib
## Joblib
from sklearn.externals import joblib
# save model
joblib.dump(xgb_model_pt2, 'FineTech_app_ML_model.joblib')
# load model
ml_model_jl = joblib.load('FineTech_app_ML_model.joblib')
# predict the output 
y_pred_jl = ml_model.predict(X_test_sc)
cm_jl = confusion_matrix(y_test, y_pred)
print('Confussion matrix = \n', cm_jl)

print("Accuracy of model = ", accuracy_score(y_test, y_pred_jl))