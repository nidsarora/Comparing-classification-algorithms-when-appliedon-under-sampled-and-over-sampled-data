_author_="nidhi"

import os
import subprocess
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from IPython.display import Image
from pyspark import SparkContext
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from  ADASYN import adasyn
style.use('ggplot')

"""
   Code to start a spark session
"""
filepath= "C:\\Users\\arora\\Documents\\nidhi\\big data project"
pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
if not "pyspark-shell" in pyspark_submit_args: pyspark_submit_args += " pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
sc =SparkContext()
print(sc)

"""
   Creating test data and training data
"""
df1=pd.DataFrame.from_csv('cs-training-processed.csv',sep=',')
df2=pd.DataFrame.from_csv('cs-training.csv',sep=',')
df_15000=df1.sample(15000)
df1_15000=df2.sample(15000)
msk = np.random.rand(len(df_15000))<0.8
training_data_sampled =df_15000[msk]
X_training_data =df_15000[msk]
test_data_sampled= df_15000[~msk]
X_test_data= df_15000[~msk]

"""
   Creating raw DataFrame with all data
"""
rawDataFrame=pd.DataFrame.from_csv('cs-training.csv',sep=',')
print(rawDataFrame.shape)
rawDataPlot=plt.plot(rawDataFrame['DebtRatio'],'.')
plt.show(rawDataPlot)
"""
   Creating raw DataFrame with removed ouliers for DebtRatio
"""
removedOutlierDebt=rawDataFrame[~(rawDataFrame["DebtRatio"] > 1)]
print(removedOutlierDebt.shape)
removedDebtOutlierPlot=plt.plot(removedOutlierDebt['DebtRatio'],'.')
plt.show(removedDebtOutlierPlot)
"""
   Creating  DataFrame with removed ouliers for 'NumberOfTime30-59DaysPastDueNotWorse'
"""
raw30DayPastPlot=plt.plot(removedOutlierDebt['NumberOfTime30-59DaysPastDueNotWorse'],'.')
plt.show(raw30DayPastPlot)
removedOutlier30to59DaysPast=removedOutlierDebt[~(removedOutlierDebt["NumberOfTime30-59DaysPastDueNotWorse"] > 8)]
removedOutlier30to59DaysPastPlot=plt.plot(removedOutlier30to59DaysPast['NumberOfTime30-59DaysPastDueNotWorse'],'.')
plt.show(removedOutlier30to59DaysPastPlot)
"""
   Creating  DataFrame with removed ouliers for "RevolvingUtilizationOfUnsecuredLines"
"""
removeOutlierUnsecuredUtilization=removedOutlier30to59DaysPast[~(removedOutlier30to59DaysPast["RevolvingUtilizationOfUnsecuredLines"] > 1)]
removeOutlierUnsecuredUtilizationPlot=plt.plot(removeOutlierUnsecuredUtilization['RevolvingUtilizationOfUnsecuredLines'],'.')
plt.show(removeOutlierUnsecuredUtilizationPlot)
print(removeOutlierUnsecuredUtilization.shape)
"""
   Creating  DataFrame with removed ouliers for Monthly Income
"""
MonthlyIncomePlot=plt.plot(removeOutlierUnsecuredUtilization['MonthlyIncome'],'.')
plt.show(MonthlyIncomePlot)
removedOutlierMonthlyIncome=removeOutlierUnsecuredUtilization[~(removeOutlierUnsecuredUtilization["MonthlyIncome"] > 300000)]
print(removedOutlierMonthlyIncome.shape)
"""
Removing the missing values
"""
df_final=removedOutlierMonthlyIncome
df_after_missing_removal=df_final[~df_final.isnull().any(axis=1)]#missing values with'NA' are removed
print(df_after_missing_removal.shape)

print(df_after_missing_removal.corr()['SeriousDlqin2yrs'])
removedOutlierMonthlyIncomePlot=plt.plot(removedOutlierMonthlyIncome['MonthlyIncome'])
plt.show(removedOutlierMonthlyIncomePlot)
print('---------------------------------')

"""
   Undersampling data
"""
df_non_defaulter=df_after_missing_removal[~(removeOutlierUnsecuredUtilization["SeriousDlqin2yrs"]==1)]
df_defauter=df_after_missing_removal[~(removeOutlierUnsecuredUtilization["SeriousDlqin2yrs"]==0)]
df_non_defaulter_sample=df_non_defaulter.sample(6582)
frames=[df_non_defaulter_sample,df_defauter]
undersampled_data=pd.concat(frames)

"""
  Preparing data for applying classification algorithms
"""
Y_train=X_training_data.pop('SeriousDlqin2yrs')
Y_test=X_test_data.pop('SeriousDlqin2yrs')
Y_test.columns=['SeriousDlqin2yrs']
X_train = pd.get_dummies(training_data_sampled)
X_test=pd.get_dummies(test_data_sampled)
X_train.columns.difference(X_test.columns)
X_test[X_train.columns.difference(X_test.columns)] = 0
X_test = X_test[X_train.columns]

"""
   Applying Decision Tree Classification Algorithm with depth 2 and 10
"""
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train,Y_train)
print("Error using depth 2 is ")
print(metrics.mean_absolute_error(Y_test, dtree.predict(X_test)))
print(metrics.mean_squared_error(Y_test, dtree.predict(X_test)))
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=10)
dtree.fit(X_train, Y_train)
print("Error using depth 10 is ")
print(metrics.mean_absolute_error(Y_test, dtree.predict(X_test)))
print(metrics.mean_squared_error(Y_test, dtree.predict(X_test)))

"""logistic_model=LogisticRegression(penalty='l2',C=1)
logistic_model.fit(X_train,Y_train)
print(metrics.accuracy_score(Y_test,logistic_model.predict(X_test)))
print(metrics.mean_absolute_error(Y_test,logistic_model.predict(X_test)))"""

"""
  Naive Bayes Classification algorithm
"""
def NaiveBayes(x):
 # df_n = df_after_missing_removal.sample(1102*x) //used when simple dataset is used
 df_n=undersampled_data.sample(131*x) #used when algorithm is applied on undersampled data
 msk = np.random.rand(len(df_n)) < 0.8
 training_data_sampled = df_n[msk]
 X_training_data = df_n[msk]
 test_data_sampled = df_n[~msk]
 X_test_data = df_n[~msk]
 Y_train = X_training_data.pop('SeriousDlqin2yrs')
 Y_test = X_test_data.pop('SeriousDlqin2yrs')
 X_train = pd.get_dummies(training_data_sampled)
 X_test = pd.get_dummies(test_data_sampled)
 X_train.columns.difference(X_test.columns)
 X_test[X_train.columns.difference(X_test.columns)] = 0
 X_test = X_test[X_train.columns]
 X_train_new, Y_train_new= oversample(X_train,Y_train)
 gnb = GaussianNB()
 y_pred = gnb.fit(X_train, Y_train).predict(X_test)
 print("Number of mislabeled points out of a total %d points : %d"
     % (X_train.shape[0],(Y_test != y_pred).sum()))
 x=metrics.precision_score(Y_test, y_pred)
 y=metrics.accuracy_score(Y_test, y_pred)
 z=metrics.recall_score(Y_test,y_pred)
 # return x,y,z
 print(metrics.classification_report(Y_test,y_pred))

# NaiveBayes(100)
"""
   Function to oversample data
"""
def oversample(X_train,Y_train): # ADASYN is a python module that implements an adaptive oversampling technique for skewed datasets.
 adsn = adasyn.ADASYN(k=7,imb_threshold=0.6, ratio=0.75)
 new_X_train, new_Y_train = adsn.fit_transform(X_train,Y_train)
 return new_X_train,new_Y_train

"""
   Function to compare various classification algorithms based on some parameters
"""
def error_formula_logistic(x):
 # df_n = df_after_missing_removal.sample(1102*x)
 df_n=undersampled_data.sample(131*x)
 msk = np.random.rand(len(df_n)) < 0.8
 training_data_sampled = df_n[msk]
 X_training_data = df_n[msk]
 test_data_sampled = df_n[~msk]
 X_test_data = df_n[~msk]
 Y_train = X_training_data.pop('SeriousDlqin2yrs')
 Y_test = X_test_data.pop('SeriousDlqin2yrs')
 X_train = pd.get_dummies(training_data_sampled)
 X_test = pd.get_dummies(test_data_sampled)
 X_train.columns.difference(X_test.columns)
 X_test[X_train.columns.difference(X_test.columns)] = 0
 X_test = X_test[X_train.columns]
 X_train_new, Y_train_new= oversample(X_train,Y_train)
 logistic_model = LogisticRegression(penalty='l2', C=1)
 result=logistic_model.fit(X_train, Y_train)
 print(logistic_model.predict(X_test))
 """
    Parameters taken are precision score,accuracy score, recall score,f1 score and support
 """
 y=metrics.precision_score(Y_test, result.predict(X_test))
 z=metrics.accuracy_score(Y_test, result.predict(X_test))
 a=metrics.recall_score(Y_test, result.predict(X_test))
 print(metrics.classification_report(Y_test, result.predict(X_test)))
 return y,z,a

 """
    Function to evaluate how performance of decision tree varies with % of data taken
 """
def error_formula_decisiontree(x):
 # df_n = df_after_missing_removal.sample(1102*x)
 df_n=undersampled_data.sample(131*x)
 msk = np.random.rand(len(df_n)) < 0.8
 training_data_sampled = df_n[msk]
 X_training_data = df_n[msk]
 test_data_sampled = df_n[~msk]
 X_test_data = df_n[~msk]
 Y_train = X_training_data.pop('SeriousDlqin2yrs')
 Y_test = X_test_data.pop('SeriousDlqin2yrs')
 X_train = pd.get_dummies(training_data_sampled)
 X_test = pd.get_dummies(test_data_sampled)
 X_train.columns.difference(X_test.columns)
 X_test[X_train.columns.difference(X_test.columns)] = 0
 X_test = X_test[X_train.columns]
 X_train_new,Y_train_new=oversample(X_train,Y_train)
 dtree = DecisionTreeClassifier(random_state=0, max_depth=2)
 dtree.fit(X_train,Y_train)
 export_graphviz(dtree, feature_names=X_train.columns)

 Image("tree.png", unconfined=True)
 dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
 model=dtree.fit(X_train,Y_train)
 pickle.dump(dtree, open("dtree.pickle", "wb"))
 reader = pickle.load(open('dtree.pickle', 'rb'))
 print(reader)
 print(metrics.mean_absolute_error(Y_test, dtree.predict(X_test)))
 print(metrics.mean_squared_error(Y_test, dtree.predict(X_test)))
 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=10,min_samples_split=5, random_state=99)
 dtree1=dtree.fit(X_train, Y_train)
 dotfile=open('C:/Users/arora/PycharmProjects/bigData/dtree2.dot','w')
 dotfile=export_graphviz(dtree1, feature_names=X_train.columns)
 dotfile.close()
 plt.plot("dot -Tpng C:/Users/arora/PycharmProjects/bigData/dtree2.dot -o C:/Users/arora/PycharmProjects/bigData/dtree2.png")

 export_graphviz(dtree, feature_names=X_train.columns)
 y = metrics.precision_score(Y_test, dtree1.predict(X_test))
 z = metrics.accuracy_score(Y_test, dtree1.predict(X_test))
 a = metrics.recall_score(Y_test, dtree1.predict(X_test))
 print(metrics.classification_report(Y_test, dtree1.predict(X_test)))
 # return y, z, a

 """
    Function to evaluate how performance of Random Forest varies with % of data taken
 """
def error_formula_randomforest(x):
 # df_n = df_after_missing_removal.sample(1102*x)
 df_n=undersampled_data.sample(131*x)
 msk = np.random.rand(len(df_n)) < 0.8
 training_data_sampled = df_n[msk]
 X_training_data = df_n[msk]
 test_data_sampled = df_n[~msk]
 X_test_data = df_n[~msk]
 Y_train = X_training_data.pop('SeriousDlqin2yrs')
 Y_test = X_test_data.pop('SeriousDlqin2yrs')
 X_train = pd.get_dummies(training_data_sampled)
 X_test = pd.get_dummies(test_data_sampled)
 X_train.columns.difference(X_test.columns)
 X_test[X_train.columns.difference(X_test.columns)] = 0
 X_test = X_test[X_train.columns]
 X_train_new,Y_train_new=oversample(X_train,Y_train)
 rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=4, max_depth=10)
 rf1=rf.fit(X_train_new,Y_train_new)
 return metrics.mean_absolute_error(Y_test, rf.predict(X_test))
 return metrics.accuracy_score(Y_test, rf1.predict(X_test))
 rf_max = RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=4, max_depth=None)
 rf_max.fit(X_train,Y_train)
 print("Random Forest Clssifier with full depth :Error is")
 print(metrics.classification_report(Y_test, rf1.predict(X_test)))

 """
    Function to draw graph based on % of input data taken to evaluate
 """
def graph(error_formula, x_range):
  x = np.array(x_range)
  y_list=[]
  z_list=[]
  a_list=[]
  for i in x:
   y,z,a=error_formula(i)
   y_list.append(y)
   z_list.append(z)
   a_list.append(a)
  print(x.shape)
  plt.figure()
  plt.plot(x, y_list,marker='o', ms = 10, alpha=1, color='b',label='Precision Score ')
  plt.plot(x, z_list,marker='o', ms = 10, alpha=1, color='r',label='Accuracy Score ')
  plt.plot(x, a_list, marker='o', ms=10, alpha=1, color='c',label='Recall Score ')
  plt.show()

graph(error_formula_logistic,range(1,100))
graph(error_formula_decisiontree,range(1,100))
graph(error_formula_randomforest,range(1,100))
graph(NaiveBayes,range(1,100))
error_formula_logistic(100)
error_formula_decisiontree(100)
error_formula_decisiontree(100)
