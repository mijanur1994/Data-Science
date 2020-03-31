# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Load The Data set
df = pd.read_csv("loan_data_set.csv", sep = ',')
df.shape

# Dropping The Duplicates
df = df.drop_duplicates()

# Show Entire Information of the data frame
df.info()

# Selecting Categorical Features
categorical_columns = ['Gender',
                       'Married',
                       'Education',
                       'Self_Employed',
                       'Property_Area',
                       'Loan_Status',
                       ]
#Looping through the columns and changing type to 'category'
for column in categorical_columns:
    df[column] = df[column].astype('category')

df.info()

# Show The data
df.head(10)
df.tail(10)

# Let's look at the distribution of the categorical data:
for feature in df.dtypes[df.dtypes == 'category'].index:
    sns.countplot(y=feature, data=df, order = df[feature].value_counts().index)
    plt.show()
    
    
#Let's now look at the number of entries per each level of the categorical variables as
# proportion of the overall number of entries:
#Type of gender as proportion of the overall number of values
df.Gender.value_counts()/df.Gender.count()

#Type of Married as proportion of the overall number of values
df.Married.value_counts()/df.Married.count()

#Type of Education as proportion of the overall number of values
df.Education.value_counts()/df.Education.count()

#Type of Self_Employed as proportion of the overall number of values
df.Self_Employed.value_counts()/df.Self_Employed.count()

#Type of Property_Area as proportion of the overall number of values
df.Property_Area.value_counts()/df.Property_Area.count()

#Type of Loan_Status as proportion of the overall number of values
df.Loan_Status.value_counts()/df.Loan_Status.count()

# =============================================================================
# Exploratory Data Analysis(Numerical_Data)
# =============================================================================
#Next, let's look at the distributions of the numerical variables:
#Histogram grid
df.hist(figsize=(10,10), color = 'red')
# Clear the text "residue"
plt.show()
#Summary of numeric features
summary_numeric = df.describe()

# =============================================================================
# Data Cleaning and Feature Engineering
# =============================================================================

#Creating a copy of the original data frame
df_copy = df.copy()

# Drpping Null values
df = df.dropna()
df.info()

# Categorical Variable Handling
X=np.where(df["Gender"]=='Male',1,0)
df['Gender']=X

X=np.where(df["Married"]=='Yes',1,0)
df['Married']=X

X=np.where(df["Education"]=='Graduate',1,0)
df['Education']=X

X=np.where(df["Self_Employed"]=='Yes',1,0)
df['Self_Employed']=X

# Dummy variable and Label_Encoding
df = pd.get_dummies(df, prefix=["Property_Area"], columns=["Property_Area"])
df["Property_Area_Urban"] = df['Property_Area_Urban'].astype(int)
df["Property_Area_Semiurban"] = df['Property_Area_Semiurban'].astype(int)

# Avoiding Dummy Variable Trap
df = df.drop(['Property_Area_Rural'], axis = 1) 
#df["Property_Area_Rural"] = df['Property_Area_Rural'].astype(int)  ###### ----(Convert Objet to int32)

# Drop Unnecessary features
df = df.drop(['Loan_ID'], axis = 1) 

# Replace Columns Values(3+) with 3
df["Dependents"] = df["Dependents"].replace('3+', '3')
df["Dependents"] = df['Dependents'].astype(int)  ###### ----(Convert Objet to int32)
#Calculate correlations between numericdf['Dependents'].value_counts()
#Calculate correlations between the features and predictor
correlations = df.corr()

#Make the figsize 7 x 6
plt.figure(figsize=(15,15))

#Plot heatmap of correlations
_ = sns.heatmap(correlations, cmap="Greens", annot=True)

# To avoid less significant columns
ax = sns.countplot(x="Gender", hue="Loan_Status", data=df)
ax = sns.countplot(x="Married", hue="Loan_Status", data=df)
ax = sns.countplot(x="Dependents", hue="Loan_Status", data=df)

# Detecting Outliers
_ = sns.pairplot(df)

# =============================================================================
# Creating Dependent and independents Variable 
# =============================================================================
column=['Dependents',
        'Education',
        'Self_Employed',
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Property_Area_Urban',
        'Property_Area_Semiurban']
x=df[column]

y=df.iloc[:,10:11].values
y=np.where(y=='Y',1,0)

# Standardization the values
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
x = standardscaler.fit_transform(x)
x= pd.DataFrame(x)

# Feature Selection Using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.80, whiten = False)
x_pca=pca.fit_transform(x)
x_pca=pd.DataFrame(x_pca)
pca.explained_variance_ratio_


'''# Plotting
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0:1],x_pca[:,0], c = y, cmap = 'plasma')
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show'''


# =============================================================================
# Fitting To Model(Logistic Regression)
# =============================================================================
# Train test splitting
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x_pca, y, test_size = 0.25, random_state = 0)

# Logistoc Regression
from sklearn.linear_model import LogisticRegression
logistic_model=LogisticRegression(class_weight='balanced',random_state=0,n_jobs=-1)
logistic_model.fit(x_train,y_train)
y_pred=logistic_model.predict(x_test)

#Evaluation of logistic model using  confusion matric
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# Check Accuracy
score = logistic_model.score(x_test, y_test)
print(score)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

########## USing K-Fold Cross Validation(Logistic Regression)###########

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

#Craete pipeline, that standarizes, then runs logistic regression
pipeline = make_pipeline(standardscaler, logistic_model)

# Create K-Fold cross validation
kf = KFold(n_splits=10,
           shuffle=True,
           random_state=0)

# Conduct k-fold
cv_result = cross_val_score(pipeline,# pipeline
                            x_train,#Feature matrix
                            y_train,# Target Vector
                            cv = kf,# Cross Validation Technique
                            scoring="accuracy",# Loss Function
                            n_jobs = -1)# Uses all CPU Scores
                            
# calculate Mean
cv_result.mean()

# =============================================================================
# Model Selection Using Grid_Search_CV(Logistic Regression)
# =============================================================================
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression()
a=np.logspace(-2, 2, 50)
param_grid = {
    'C': np.logspace(-2, 2, 50),
    'penalty': ['l1','l2'],
    'class_weight': [None,'balanced'],
    'max_iter' : np.arange(10,100,1)
    
}
logistic = LogisticRegression()
grid_search = GridSearchCV(estimator = logistic, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 10)

grid_search.fit(x_train,y_train)
grid_search.best_params_

#after grid search
logistic = LogisticRegression(C = 0.01,
                              class_weight = None, 
                              max_iter = 10, 
                              penalty = 'l2')
logistic.fit(x_train,y_train)
y_pred_glr=logistic.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_glr=confusion_matrix(y_test,y_pred_glr)

# Accuracy after using Grid_Search
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_glr))

# AUC ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix, classification_report
#Obtaining the ROC score
roc_auc = roc_auc_score(y_test, y_pred_glr)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, logistic.predict_proba(x_test)[:,1])
plt.plot(fpr, tpr, label='L2 Logistic Regression (area = %0.03f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve for Logistic regression')
plt.legend(loc="upper left")

# =============================================================================
# Fitting Model(Decesion tree)
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
decesion_tree = DecisionTreeClassifier()
decesion_tree.fit(x_train, y_train)
y_pred_dt = decesion_tree.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dt=confusion_matrix(y_test,y_pred_dt)

# Check Accuracy
score_dt = decesion_tree.score(x_test, y_test)
print(score_dt)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_dt))

# =============================================================================
# Fitting Model(Naive Bayes)
# =============================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train, y_train)

# Predicting the Test set results
y_pred_nb = Naive_bayes.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_nb))

# =============================================================================
# Fitting Model(Support Vector Classification)
# =============================================================================
#SVM(Support_Vector_Machine)
# Train test splitting
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.25, random_state = 0)

# Creating SVM with radial Kernel(Gaussian Kernel or RBF)
from sklearn.svm import SVC
model_rbf = SVC(kernel = 'rbf', random_state = 42)
model_rbf.fit(x_train,y_train)

#Check Performance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = model_rbf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, model_rbf.predict(x_test)))


# Creating SVM with Linear Kernel
model_lin = SVC(kernel = 'linear', random_state=42)
model_lin.fit(x_train,y_train)
y_pred_lin = model_lin.predict(x_test)
print(accuracy_score(y_test, y_pred_lin))
print(confusion_matrix(y_test, y_pred_lin))
print(classification_report(y_test, model_lin.predict(x_test)))

#Check Performance of linear kernel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred_lin = model_lin.predict(x_test)
print(accuracy_score(y_test, y_pred_lin))
print(confusion_matrix(y_test, y_pred_lin))
print(classification_report(y_test, model_lin.predict(x_test)))


# Creating SVM with Polynomial Kernel
model_poly = SVC(kernel = 'poly',gamma = .1, C = 0.8, degree = 2)
model_poly.fit(x_train, y_train)
y_pred_poly = model_poly.predict(x_test)
print(accuracy_score(y_test, y_pred_poly))
print(confusion_matrix(y_test, y_pred_poly))
print(classification_report(y_test, model_poly.predict(x_test)))

#Check Performance of polynomial kernel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred_poly = model_poly.predict(x_test)
print(accuracy_score(y_test, y_pred_poly))
print(confusion_matrix(y_test, y_pred_poly))
print(classification_report(y_test, model_poly.predict(x_test)))


# =============================================================================
# Fitting Model(Random Forest)
# =============================================================================
# Train test splitting
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 15,#number of trees,
                                       bootstrap=False,#sampling with replacement or without replacement,
                                       criterion="gini",
                                       max_depth=None,
                                       max_features="auto",
                                       verbose=10,
                                       class_weight='balanced',
                                       random_state=0,
                                       n_jobs=-1)

random_forest.fit(x_train, y_train)
y_pred_rf = random_forest.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf=confusion_matrix(y_test,y_pred_rf)

# Check Accuracy
score_rf = random_forest.score(x_test, y_test)
print(score_rf)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))


# =============================================================================
# Model Selection Using Grid_Search_CV(Random_Forest)
# =============================================================================
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': np.arange(1,10,1),
    'max_features': [2, 3, 5,],
    'min_samples_leaf': [3, 4, 5, 6, 7],
    'min_samples_split': [8, 10, 12, 14],
    'n_estimators': np.arange(10, 100, 5)
}
random_forest = RandomForestClassifier()
grid_search = GridSearchCV(estimator = random_forest, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 10)

grid_search.fit(x_train,y_train)
grid_search.best_params_


#  After using grid_search fitting best parameter to Random Forest
random_forest = RandomForestClassifier(bootstrap = True,
                                       max_depth = 9,
                                       max_features = 2,
                                       min_samples_leaf = 3,
                                       min_samples_split = 8,
                                       n_estimators = 20)

random_forest.fit(x_train,y_train)
y_pred_grf=random_forest.predict(x_test)

# Confusion Matrix after using Grid_Search
from sklearn.metrics import confusion_matrix
cm_grf=confusion_matrix(y_test,y_pred_grf)

# Accuracy after using Grid_Search
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_grf))

# AUC ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix, classification_report
#Obtaining the ROC score
roc_auc = roc_auc_score(y_test, y_pred_grf)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, random_forest.predict_proba(x_test)[:,1])
plt.plot(fpr, tpr, label='L2 random forest (area = %0.03f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve for random forest')
plt.legend(loc="upper left")

