import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
plt.rcParams['figure.figsize'] = (8, 7)
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#preprocessing dataset hungarian
dataset1=pd.read_csv('hungarian.data')
dataset_string1=dataset1.to_string()
dataset_string1.replace("\r",' ')
datalist1 = list(dataset_string1.split('name'))

for i in datalist1:
    res1 = [val for idx, val in enumerate(datalist1)
           if val or (not val and datalist1[idx - 1])]
while(" " in res1):
    res1.remove(" ")

df1=pd.DataFrame(res1,columns=['col1'])
df1=df1.col1.str.split(expand=True)
df1.dropna(axis=1,how='any')
for i in range(0,81,8):
    df1.drop([i],axis=1,inplace=True)

#preprocessing dataset long -beach-va

dataset2=pd.read_csv('long-beach-va.data')
dataset_string2=dataset2.to_string()
dataset_string1.replace("\r",' ')
datalist2 = list(dataset_string2.split('name'))
for i in datalist2:
    res2 = [val for idx, val in enumerate(datalist2)
           if val or (not val and datalist2[idx - 1])]
while(" " in res2):
    res2.remove(" ")

df2=pd.DataFrame(res1,columns=['col1'])
df2=df2.col1.str.split(expand=True)
df2.dropna(axis=1,how='any')
for i in range(0,81,8):
    df2.drop([i],axis=1,inplace=True)

#preprocessing  dataset switzerland

dataset3=pd.read_csv('switzerland.data')
dataset_string3=dataset3.to_string()
dataset_string3.replace("\r",' ')
datalist3 = list(dataset_string3.split('name'))
print(len(datalist3))
for i in datalist3:
    res3 = [val for idx, val in enumerate(datalist3)
           if val or (not val and datalist3[idx - 1])]
while(" " in res3):
    res3.remove(" ")

df3=pd.DataFrame(res3,columns=['col1'])
df3=df3.col1.str.split(expand=True)
df3.dropna(axis=1,how='any')
for i in range(0,81,8):
    df3.drop([i],axis=1,inplace=True)

#preprocessing  cleveland
"""file1=open('cleveland.data',errors='ignore')
print(file1)
dataset4=pd.read_csv('cleveland.data')
dataset_string4=dataset4.to_string()
dataset_string4.replace("\r",' ')
datalist4 = list(dataset_string4.split('name'))
print(len(datalist4))
for i in datalist4:
    res4 = [val for idx, val in enumerate(datalist4)
           if val or (not val and datalist4[idx - 1])]
while(" " in res4):
    res4.remove(" ")

df4=pd.DataFrame(res4,columns=['col1'])
df4=df4.col1.str.split(expand=True)
df4.dropna(axis=1,how='any')
for i in range(0,81,8):
    df4.drop([i],axis=1,inplace=True)
"""
#print(df1)
#print(df2)
#print(df3)
df=df1.append(df2,ignore_index=True)
df=df.append(df3,ignore_index=True)
df=pd.DataFrame(df)
#print(df.dtypes)
df.to_csv('old.csv')
df.replace(to_replace=r'^-9.$',value='None',regex=True,inplace=True)
df.replace(to_replace=r'^-9$',value='None',regex=True,inplace=True)
def missing(dff):
    miss = dff.isnull().sum() / len(dff)
    miss = miss[miss > 0]
    miss.sort_values(inplace=True)
    print(miss)
    miss = miss.to_frame()
    miss.columns = ['count']
    miss.index.names = ['Name']
    miss['Name'] = miss.index
    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x='Name', y='count', data=miss)
    plt.xticks(rotation=90)
    plt.show()

missing(df)
def plot_graphs(dfnew):
    plt.matshow(temp_corr)
    plt.title('Correlation Matrix', fontsize=16)
    cb = plt.colorbar()
    plt.xticks(range(dfnew.shape[1]), df.columns, fontsize=12)
    plt.yticks(range(dfnew.shape[1]), df.columns, fontsize=12)
    plt.show()
    sns.set_context(font_scale=2, rc={"font.size": 5, "axes.titlesize": 25, "axes.labelsize": 20})
    sns.catplot(kind='count', data=dfnew, x=3, hue='target', order=dfnew[3].sort_values().unique())
    plt.title('Variation of age with each class', fontsize=12)
    plt.show()

    sns.catplot(kind='count', data=dfnew, x=11, hue='target', order=dfnew[11].sort_values().unique())
    plt.title('variation of age and testtrcpcs',fontsize=12)
    plt.show()

    sns.catplot(kind='count', data=dfnew, x=10, hue='target', order=dfnew[10].sort_values().unique())
    plt.title('variation of heart disease with different chest pain',fontsize=12)
    plt.show()
    sns.catplot(kind='count', data=dfnew, x=36, hue='target', order=dfnew[36].sort_values().unique())
    plt.title('variation of age and maximum heart rate acheived',fontsize=12)
    plt.show()
    sns.catplot(kind='count', data=dfnew, x=37, hue='target', order=dfnew[37].sort_values().unique())
    plt.title('variation of age and resting heart rate',fontsize=12)
    plt.show()
    sns.catplot(kind='count', data=dfnew, x=38, hue='target', order=dfnew[38].sort_values().unique())
    plt.title('variation of age and peak exercise blood pressure',fontsize=12)
    plt.show()
    sns.catplot(kind='count', data=dfnew, x=39, hue='target', order=dfnew[39].sort_values().unique())
    plt.title('variation of age and peak exercise blood pressure',fontsize=12)
    plt.show()


df.dropna(axis=1,how='all',inplace=True)
df.dropna(axis=0,how='any',inplace=True)
df = df.loc[:, (df.isin([' ', 'NULL','','None','-9.0','-9','NaN']) | df.isnull()).mean() < .01];
liscol=[]
#print(df.head())
for i in range (1,47,1):
    liscol.append(i)
#dfnew=pd.DataFrame(df,columns=liscol)

for column in df.columns:
    df[column].fillna(df[column].mode(), inplace=True)
df.drop([1,71,62,53,26,35,44,17],axis=1,inplace=True)
#df.fillna(0)
df.drop([2,25],axis=1,inplace=True)
df.fillna(method='pad',inplace=True)
df.mask(df.astype(object).eq(None)).dropna(inplace=True)
dfnew=df.replace(to_replace='None',value=np.nan).dropna()
#print(df)

dfnew.to_csv('heart_dataset.csv')

print(dfnew)

Y1=pd.DataFrame(dfnew[65])
dftemp=pd.DataFrame(dfnew)
dftemp=dfnew.drop([65],axis=1)
X=pd.DataFrame(dftemp)
Y=Y1.replace(to_replace=['1','2','3','4'],value=['1','1','1','1'])
Y=pd.DataFrame(Y)
print(Y)
dfnew['target']=Y
print(dfnew)
temp_corr=dfnew.astype(float).corr()
#print(temp_corr)
print(dfnew.info())
print(dfnew)
plot_graphs(dfnew)

#APPLYING Various Models for accuracy
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model1=LogisticRegression()
model=model1.fit(X_train,Y_train)
pred=model1.predict(X_test)
#print(pred)
score1=model1.score(X_test,Y_test)
scoreslist_decisiontree=[]
depth_list=[]
for k in range(2,30,2):
    depth_list.append(k)
    model2=DecisionTreeClassifier(max_depth=k,random_state=0)
    model2.fit(X_train, Y_train)
    score2=model2.score(X_test,Y_test)
    scoreslist_decisiontree.append(score2)
model3=GaussianNB()
model3.fit(X_train, Y_train)
score3=model3.score(X_test,Y_test)
k_list=[]
kneighbour_list=[]
for k in range(1,20,1):
    k_list.append(k)
    model4=KNeighborsClassifier(n_neighbors=k)
    model4.fit(X_train, Y_train)
    score4=model4.score(X_test,Y_test)
    kneighbour_list.append(score4)
print(score1)
print(score2)
print(score3)
print(score4)
print(scoreslist_decisiontree)
plt.plot(depth_list,scoreslist_decisiontree,label='accuracy score with different depths of the decision tree')
plt.legend()
plt.show()
model5=SVC(kernel='linear')
model5.fit(X_train,Y_train)
score5=model5.score(X_test,Y_test)
print(score5)
plt.plot(k_list,kneighbour_list,label='accuracy score with varing values of neighbours')
plt.legend()
plt.show()
scorelist_classifiers=[]
x_label=['Linear Regression','Decision Tree','GaussianNB','KNN','SVM']
scorelist_classifiers.append(score1)
scorelist_classifiers.append(score2)
scorelist_classifiers.append(score3)
scorelist_classifiers.append(score4)
scorelist_classifiers.append(score5)
plt.bar(x_label,scorelist_classifiers)
plt.xlabel("classifiers")
plt.ylabel("Accuracy")
plt.show()

