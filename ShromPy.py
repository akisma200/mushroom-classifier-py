# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:44:10 2018

@author: AKISMA200
"""

import time
import pandas as pd
import numpy as np
import sklearn as sk

import pickle

# Some modules for plotting and visualizing
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# And some Machine Learning modules from scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV





#read mushroom.csv
shroomdata = pd.read_csv('C:\\Users\\AKISMA200\\Downloads\\Telegram Desktop\\mushroom.csv')

#show data
print(shroomdata.head())



#show data details
print(shroomdata.describe())

#show catagories
for col in shroomdata.columns.values:
    print(col, shroomdata[col].unique())

#show rows and columns
print(shroomdata.shape)

#show where null values are
print(shroomdata.isnull().sum())

#drop all null values because there aren't enought to seriously dent the project
print(shroomdata.dropna(inplace = True))

#show data with dropped values
print(shroomdata.head())

#remove any columns that dont affect the outcome (if an attribute is global then it doesn not affect decision making)
for col in shroomdata.columns.values:
    if len(shroomdata[col].unique()) <= 1:
        print("Removing column {}, which only contains the value: {}".format(col, shroomdata[col].unique()[0]))
        shroomdata.drop('veil-type',axis=1,inplace=True)

#expand categories into "IS" booleans
y = shroomdata['Class']
Y_enc = pd.get_dummies(y)
print('this is the mushroom Class being encoded')
print(Y_enc.head())
new_Y_enc = Y_enc.iloc[:,0]
print('this is the mushroom Class after the Class_is_p is removed')
print(new_Y_enc.head())
x = shroomdata.iloc[:,0:20]
X_enc = pd.get_dummies(x)
print(X_enc.head())

#encode the catagorical data
new_data = shroomdata
for i in shroomdata.columns:
    new_data[i] = LabelEncoder().fit_transform(shroomdata[i])
print(new_data.head())

# Train Test Split
X = new_data.drop('Class', axis=1)
y = new_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

keys = []
scores = []
models = {'rbf SVM':SVC(kernel='rbf', gamma=.10, C=1.0), 'linear SVM':SVC(kernel='linear', gamma=.10, C=1.0), 'sigmoid':SVC(kernel='sigmoid', gamma=.10, C=1.0) }

for k,v in models.items():
    mod = v
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test)
    print('Results for: ' + str(k) + '\n')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    acc = accuracy_score(y_test, pred)
    print(acc)
    print('\n' + '\n')
    keys.append(k)
    scores.append(acc)
    table = pd.DataFrame({'model':keys, 'accuracy score':scores})

print(table)

#display_corr_with_col(shroomdata, 'Class')
# Train Test Split
X = X_enc
y = new_Y_enc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

keys = []
scores = []
models = {'rbf SVM':SVC(kernel='rbf', gamma=.10, C=1.0), 'linear SVM':SVC(kernel='linear', gamma=.10, C=1.0), 'sigmoid':SVC(kernel='sigmoid', gamma=.10, C=1.0)}

for k,v in models.items():
    mod = v
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test)
    print('Results for: ' + str(k) + '\n')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    acc = accuracy_score(y_test, pred)
    print(acc)
    print('\n' + '\n')
    keys.append(k)
    scores.append(acc)
    table = pd.DataFrame({'model':keys, 'accuracy score':scores})

print(table)

#display_corr_with_col(X_enc, new_Y_enc)
#confutable
#svc=SVC(C=0.10,gamma=0.001,kernel='rbf')
#svc.fit(X_train,y_train)
#svc.score(X_test,y_test)
#Ypreds=svc.predict(X_test)
#cm = confusion_matrix(y_test,Ypreds)
#xy=np.array([0,1])
#plt.figure(figsize=(10,10))
#sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',xticklabels=xy,yticklabels=xy, fmt= 'g')

#svc=SVC()
#param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],'C': [1, 10, 100, 1000]},
#              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]##
#
#grid = GridSearchCV(svc,param_grid,cv=5,scoring='accuracy')
#print("Tuning hyper-parameters")
#grid.fit(X_train, y_train)
#print(grid.best_params_)
#print(np.round(grid.best_score_,3))


ax = sns.pairplot(shroomdata, hue='Class')
plt.title('Pairwise relationships between the features')
plt.show()

#ax = sns.pairplot(shroomdata, hue='odor')
#plt.title('Pairwise relationships between the features')
#plt.show()

	
X = shroomdata.values
X_std = StandardScaler().fit_transform(X)
 
pca = PCA().fit(X_std)
var_ratio = pca.explained_variance_ratio_
components = pca.components_
print(pca.explained_variance_)
plt.plot(np.cumsum(var_ratio))
plt.xlim(0,9,1)
plt.xlabel('Number of Attributes', fontsize=16)
plt.ylabel('Cumulative explained variance', fontsize=16)
plt.show()


	
correlation_matrix = shroomdata.corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
plt.title('Correlation matrix between the attributes', fontsize=20)
plt.show()

#new_data.drop('gill-size',axis=1,inplace=True)
#new_data.drop('habitat',axis=1,inplace=True)
#new_data.drop('population',axis=1,inplace=True)
#new_data.drop('veil-color',axis=1,inplace=True)
#new_data.drop('gill-attachment',axis=1,inplace=True)
#new_data.drop('cap-surface',axis=1,inplace=True)
#new_data.drop('cap-shape',axis=1,inplace=True)
#new_data.drop('cap-color',axis=1,inplace=True)
#new_data.drop('gill-spacing',axis=1,inplace=True)
#new_data.drop('spore-print-color',axis=1,inplace=True)
#new_data.drop('stalk-shape',axis=1,inplace=True)
#new_data.drop('stalk-surface-above-ring',axis=1,inplace=True)
#new_data.drop('stalk-surface-below-ring',axis=1,inplace=True)
#new_data.drop('stalk-color-above-ring',axis=1,inplace=True)
#new_data.drop('stalk-color-below-ring',axis=1,inplace=True)

#X = new_data.drop('Class', axis = 1)
#y = new_data['Class']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#keys = []
##scores = []
#models = {'rbf SVM':SVC(kernel='rbf', gamma=.10, C=1.0), 'linear SVM':SVC(kernel='linear', gamma=.10, C=1.0), 'sigmoid':SVC(kernel='sigmoid', gamma=.10, C=1.0)}
#
#for k,v in models.items():
 #   mod = v
 ##   mod.fit(X_train, y_train)
 #   pred = mod.predict(X_test)
 #   print('Results for: ' + str(k) + '\n')
 #   print(confusion_matrix(y_test, pred))
 #   print(classification_report(y_test, pred))
 #   acc = accuracy_score(y_test, pred)
 #   print(acc)
 #   print('\n' + '\n')
 #    keys.append(k)
 #   scores.append(acc)
 #   table = pd.DataFrame({'model':keys, 'accuracy score':scores})
#
##print(table)



#sns.countplot(x = 'odor', shroomdata = data, hue='class', palette='coolwarm')#
#plt.show()


############################################## ONE WORKING SOLUTION BUT ITS SHITE #########################################
#from sklearn.svm import SVC
#svm_model= SVC()
#
#train , test = train_test_split(new_data, test_size = 0.3)
#train_y = train['Class']
#train_x = train[[x for x in train.columns if 'Class' not in x]]
#
#test_y = test['Class']
#test_x = test[[x for x in test.columns if 'Class' not in x]]
#
#models = [SVC(kernel='rbf', random_state=0), SVC(kernel='linear', random_state=0)]
#model_names = ['SVC_rbf', 'SVC_linear']
#for i, model in enumerate(models):
#    model.fit(train_x, train_y)
 #   print('The accurancy of ' + model_names[i] + ' is ' + str(accuracy_score(test_y, model.predict(test_x))) )
###########################################################################################################################
    
sns.countplot(x='Class',data=shroomdata)


	
def display_corr_with_col(df, col):
    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('The Correlation of All Attributes With {}'.format(col), fontsize=20)
    ax.set_ylabel('Correlation Coefficient Value', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()

display_corr_with_col(new_data, 'Class')
===========================================================================================
===========================================================================================
============================BELOW IS THE EXPERIMENTAL TEMPLATE=============================
============================CHOP THIS UP HOWEVER YOU LIKE==================================
===========================================================================================
# Lets import some modules for basic computation
import time
import pandas as pd
import numpy as np

import pickle

# Some modules for plotting and visualizing
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# And some Machine Learning modules from scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#These Classifiers have been commented out because they take too long and do not give more accuracy as the other ones.
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.gaussian_process import GaussianProcessClassifier




dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    #"AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process": GaussianProcessClassifier()
}

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.
    
    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train. 
    So it is best to train them on a smaller dataset first and 
    decide whether you want to comment them out or not based on the test accuracy score.
    """
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models

def label_encode(df, list_columns):
    """
    This method one-hot encodes all column, specified in list_columns
    
    """
    for col in list_columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)

        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col] = col_values_transformed      

def expand_columns(df, list_columns):
    for col in list_columns:
        colvalues = df[col].unique()
        for colvalue in colvalues:
            newcol_name = "{}_is_{}".format(col, colvalue)
            df.loc[df[col] == colvalue, newcol_name] = 1
            df.loc[df[col] != colvalue, newcol_name] = 0
    df.drop(list_columns, inplace=True, axis=1)
        
def get_train_test(df, y_col, x_cols, ratio):
    """ 
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test

def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))

def display_corr_with_col(df, col):
    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()

df_mushrooms = 'C:\Users\AKISMA200\Downloads\Telegram Desktop\mushroom.csv'
df_mushrooms = pd.read_csv(df_mushrooms)
display(df_mushrooms.head())

for col in df_mushrooms.columns.values:
    print(col, df_mushrooms[col].unique())


for col in df_mushrooms.columns.values:
    if len(df_mushrooms[col].unique()) <= 1:
        print("Removing column {}, which only contains the value: {}".format(col, df_mushrooms[col].unique()[0]))




print("Number of rows in total: {}".format(df_mushrooms.shape[0]))
print("Number of rows with missing values in column 'cap-shape': {}".format(df_mushrooms[df_mushrooms['cap-shape'] == '??'].shape[0]))
df_mushrooms_dropped_rows = df_mushrooms[df_mushrooms['cap-shape'] != '??']


drop_percentage = 0.8

df_mushrooms_dropped_cols = df_mushrooms.copy(deep=True)
df_mushrooms_dropped_cols.loc[df_mushrooms_dropped_cols['cap-shape'] == '?', 'cap-shape'] = np.nan

for col in df_mushrooms_dropped_cols.columns.values:
    no_rows = df_mushrooms_dropped_cols[col].isnull().sum()
    percentage = no_rows / df_mushrooms_dropped_cols.shape[0]
    if percentage > drop_percentage:
        del df_mushrooms_dropped_cols[col]
        print("Column {} contains {} missing values. This is {} percent. Dropping this column.".format(col, no_rows, percentage))


df_mushrooms_zerofill = df_mushrooms.copy(deep = True)
df_mushrooms_zerofill.loc[df_mushrooms_zerofill['cap-shape'] == '?', 'cap-shape'] = np.nan
df_mushrooms_zerofill.fillna(0, inplace=True)

df_mushrooms_bfill = df_mushrooms.copy(deep = True)
df_mushrooms_bfill.loc[df_mushrooms_bfill['cap-shape'] == '?', 'cap-shape'] = np.nan
df_mushrooms_bfill.fillna(method='bfill', inplace=True)

df_mushrooms_ffill = df_mushrooms.copy(deep = True)
df_mushrooms_ffill.loc[df_mushrooms_ffill['cap-shape'] == '?', 'cap-shape'] = np.nan
df_mushrooms_ffill.fillna(method='ffill', inplace=True)

df_mushrooms_ohe = df_mushrooms.copy(deep=True)
to_be_encoded_cols = df_mushrooms_ohe.columns.values
label_encode(df_mushrooms_ohe, to_be_encoded_cols)
display(df_mushrooms_ohe.head())

## Now lets do the same thing for the other dataframes
df_mushrooms_dropped_rows_ohe = df_mushrooms_dropped_rows.copy(deep = True)
df_mushrooms_zerofill_ohe = df_mushrooms_zerofill.copy(deep = True)
df_mushrooms_bfill_ohe = df_mushrooms_bfill.copy(deep = True)
df_mushrooms_ffill_ohe = df_mushrooms_ffill.copy(deep = True)

label_encode(df_mushrooms_dropped_rows_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_zerofill_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_bfill_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_ffill_ohe, to_be_encoded_cols)


y_col = 'Class'
to_be_expanded_cols = list(df_mushrooms.columns.values)
to_be_expanded_cols.remove(y_col)

df_mushrooms_expanded = df_mushrooms.copy(deep=True)
label_encode(df_mushrooms_expanded, [y_col])
expand_columns(df_mushrooms_expanded, to_be_expanded_cols)
display(df_mushrooms_expanded.head())

## Now lets do the same thing for all other dataframes
df_mushrooms_dropped_rows_expanded = df_mushrooms_dropped_rows.copy(deep = True)
df_mushrooms_zerofill_expanded = df_mushrooms_zerofill.copy(deep = True)
df_mushrooms_bfill_expanded = df_mushrooms_bfill.copy(deep = True)
df_mushrooms_ffill_expanded = df_mushrooms_ffill.copy(deep = True)

label_encode(df_mushrooms_dropped_rows_expanded, [y_col])
label_encode(df_mushrooms_zerofill_expanded, [y_col])
label_encode(df_mushrooms_bfill_expanded, [y_col])
label_encode(df_mushrooms_ffill_expanded, [y_col])

expand_columns(df_mushrooms_dropped_rows_expanded, to_be_expanded_cols)
expand_columns(df_mushrooms_zerofill_expanded, to_be_expanded_cols)
expand_columns(df_mushrooms_bfill_expanded, to_be_expanded_cols)
expand_columns(df_mushrooms_ffill_expanded, to_be_expanded_cols)


dict_dataframes = {
    "df_mushrooms_ohe": df_mushrooms_ohe,
    "df_mushrooms_dropped_rows_ohe": df_mushrooms_dropped_rows_ohe,
    "df_mushrooms_zerofill_ohe": df_mushrooms_zerofill_ohe,
    "df_mushrooms_bfill_ohe": df_mushrooms_bfill_ohe,
    "df_mushrooms_ffill_ohe": df_mushrooms_ffill_ohe,
    "df_mushrooms_expanded": df_mushrooms_expanded,
    "df_mushrooms_dropped_rows_expanded": df_mushrooms_dropped_rows_expanded,
    "df_mushrooms_zerofill_expanded": df_mushrooms_zerofill_expanded,
    "df_mushrooms_bfill_expanded": df_mushrooms_bfill_expanded,
    "df_mushrooms_ffill_expanded": df_mushrooms_ffill_expanded
}


y_col = 'Class'
train_test_ratio = 0.7

for df_key, df in dict_dataframes.items():
    x_cols = list(df.columns.values)
    x_cols.remove(y_col)
    df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, y_col, x_cols, train_test_ratio)
    dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8, verbose=False)
    
    print()
    print(df_key)
    display_dict_models(dict_models)
    print("-------------------------------------------------------")
    
    
    
    
    

	
