#!/usr/bin/env python
# coding: utf-8

# ### Import the necessary libraries

# In[1]:


import pandas as pd


# ### reading the csv file

# In[2]:


data=pd.read_csv("C://Users//user//Downloads//WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.describe()


# ### checking for duplicate values and dropping if any

# In[8]:


data.duplicated().value_counts()


# In[9]:


data=data.drop_duplicates()
data.shape


# ### checking for null values and percentage of null values

# In[10]:


data.isnull().sum()


# In[11]:


data.isnull().sum()/len(data)*100


# ### there are no duplicate values and no null values in the given dataset so we will move to the outliers 

# ### checking the outliers 

# In[12]:


import seaborn as sns


# In[13]:


sns.boxplot(data)


# In[14]:


sns.boxplot(data["SeniorCitizen"])


# ### checking the unique elements of object,float,int datatype in dataset

# In[15]:


for col in data.columns:
    if data[col].dtype == 'object':
        print(f'{col} : {data[col].unique()}')


# In[16]:


for col in data.columns:
    if data[col].dtype == 'float':
        print(f'{col} : {data[col].unique()}')


# In[17]:


for col in data.columns:
    if data[col].dtype == 'int64':
        print(f'{col} : {data[col].unique()}')


# ### as there is no use of gender and customer id i am dropping the columns from the data  

# In[18]:


data=data.drop(columns=['gender','customerID'])
data.columns
data.shape


# ### data visualisation

# In[19]:


import matplotlib.pyplot as plt
variables_to_plot = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


n_cols = 3
n_rows = (len(variables_to_plot) + n_cols - 1) // n_cols  
plt.figure(figsize=(15, n_rows * 4)) 
for i, variable in enumerate(variables_to_plot, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(x=variable, hue='Churn', data=data, palette='viridis')
    plt.title(f'Churn by {variable}')
    plt.xlabel('')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Churn', loc='upper right')

plt.tight_layout()


plt.show()


# In[20]:


num=data.select_dtypes(include='number')


# In[21]:


num.head()


# In[22]:


figure,axes=plt.subplots(3,3,figsize=(15,20))
i=0
color=["#FF0000", "#0000FF", "#008000", "#FFFF00", "#800080", "#FFA500", "#00FFFF", "#FF00FF"]
for col in num:
    sns.distplot(data[col],hist=False,ax=axes[i][0],color=color[i])
    sns.boxplot(data[col],orient='h',ax=axes[i][1],color=color[i])
    sns.histplot(data[col],ax=axes[i][2],color=color[i])
    i=i+1


# In[23]:


cat=data.select_dtypes(include='O')


# In[24]:


cat.head()


# In[25]:


cat.columns


# ### i have divided the dataset into categorical and numerical and named them as cat and num

# ### encoding

# In[26]:


get_ipython().system('pip install feature-engine -q')


# In[27]:


import feature_engine


# In[28]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Churn'] = encoder.fit_transform(data['Churn'])


# In[29]:


b=data["Churn"]
b


# In[30]:


# Apply LabelEncoder to each categorical column
for col in cat:
    data[col] = encoder.fit_transform(data[col])

data.head()


# In[31]:


data.columns


# ### scaling 

# In[32]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
data=data.drop(['Churn'],axis=1)
data=mms.fit_transform(data)
data=pd.DataFrame(data)


# In[33]:


data.head()


# In[34]:


data.columns=['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges']


# In[35]:


data.head()


# ### now the data is ready to fit into the model 

# ## machine learning classification models 

# ### splitting the data into train and test 

# In[36]:


x=data
y=b


# In[37]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42,stratify=y)


# In[38]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### logistic regression 

# In[39]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


# In[40]:


y_train_pred = log_reg.predict(x_train)


# In[41]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_train, y_train_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_train, y_train_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_train, y_train_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_train, y_train_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_train, y_train_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_train, y_train_pred)
print('classification report :', cls_rep)


# ### Logistic Regression training accuracy is 80

# In[42]:


y_test_pred = log_reg.predict(x_test)


# In[43]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_test, y_test_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_test, y_test_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_test, y_test_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_test, y_test_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_test, y_test_pred)
print('classification report :', cls_rep)


# ### Logistic Regression testing  accuracy is 79

# ### decision tree classifier

# In[44]:


# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 2)

# fit the model
clf_gini.fit(x_train,y_train)


# In[45]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

grid_search = GridSearchCV(estimator=clf_gini, param_grid=param_grid, scoring=scoring_metrics, refit='recall', cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best recall score found: ", grid_search.best_score_)


# In[46]:


y_train_pred = clf_gini.predict(x_train)


# In[47]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_train, y_train_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_train, y_train_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_train, y_train_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_train, y_train_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_train, y_train_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_train, y_train_pred)
print('classification report :', cls_rep)


# ### decision tree training accuracy is 82

# In[48]:


y_test_pred = clf_gini.predict(x_test)


# In[49]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_test, y_test_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_test, y_test_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_test, y_test_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_test, y_test_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_test, y_test_pred)
print('classification report :', cls_rep)


# ### decision tree testing accuracy is 77

# ### Random forest classifier 

# In[50]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(criterion= 'gini', max_depth= 8, max_features= 'sqrt', n_estimators= 100)
RFC.fit(x_train,y_train)


# In[51]:


y_train_pred = RFC.predict(x_train)


# In[52]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_train, y_train_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_train, y_train_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_train, y_train_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_train, y_train_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_train, y_train_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_train, y_train_pred)
print('classification report :', cls_rep)


# ### random forest train accuracy is 84

# In[53]:


y_test_pred = RFC.predict(x_test)


# In[54]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_test, y_test_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_test, y_test_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_test, y_test_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_test, y_test_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_test, y_test_pred)
print('classification report :', cls_rep)


# ### random forest testing accuracy is 80

# ### svc 

# In[55]:


from sklearn.svm import SVC
import numpy as np
svc_model = SVC(kernel='rbf',degree = 3, C=1.0, gamma='scale')
svc_model.fit(x_train,y_train)


# In[56]:


y_train_pred = svc_model.predict(x_train)


# In[57]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_train, y_train_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_train, y_train_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_train, y_train_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_train, y_train_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_train, y_train_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_train, y_train_pred)
print('classification report :', cls_rep)


# ### svc training accuracy is 81

# In[58]:


y_test_pred = svc_model.predict(x_test)


# In[59]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_test, y_test_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_test, y_test_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_test, y_test_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_test, y_test_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_test, y_test_pred)
print('classification report :', cls_rep)


# ### svc testing accuracy is 79

# ### XGB CLASSIFIER

# In[60]:


from xgboost import XGBClassifier
xgb_clf = XGBClassifier(maxdepth=4,learning_rate=1.0,n_estimators=100)
xgb_clf.fit(x_train,y_train)


# In[61]:


y_train_pred = xgb_clf.predict(x_train)


# In[62]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_train, y_train_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_train, y_train_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_train, y_train_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_train, y_train_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_train, y_train_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_train, y_train_pred)
print('classification report :', cls_rep)


# ### XGB CLASSIFIER training accuracy is 99

# In[63]:


y_test_pred = xgb_clf.predict(x_test)


# In[64]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_test, y_test_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_test, y_test_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_test, y_test_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_test, y_test_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_test, y_test_pred)
print('classification report :', cls_rep)


# ### xgb classifier testing accuracy is 76

# ### ada boost classifier

# In[65]:


from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(n_estimators=100,learning_rate=0.01,random_state=0)
model1 =abc.fit(x_train,y_train)


# In[66]:


y_train_pred = abc.predict(x_train)


# In[67]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_train, y_train_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_train, y_train_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_train, y_train_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_train, y_train_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_train, y_train_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_train, y_train_pred)
print('classification report :', cls_rep)


# ### adaboost training accuracy is 73

# In[68]:


y_test_pred = abc.predict(x_test)


# In[69]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
acc_score = accuracy_score(y_test, y_test_pred)
print('Accuracy score :', acc_score)
pre_score = precision_score(y_test, y_test_pred, average='weighted')
print('Precision score :', pre_score)
re_call = recall_score(y_test, y_test_pred, average='weighted')
print('recall score :' ,re_call)
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('f1 score :',f1_score )
con_mat = confusion_matrix(y_test, y_test_pred)
print('confusion matrix :', con_mat)
cls_rep = classification_report(y_test, y_test_pred)
print('classification report :', cls_rep)


# ### adaboost testing accuracy is 73

# ### cross validation

# In[70]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold


# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Set up K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Alternatively, use StratifiedKFold to maintain the percentage of samples for each class
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_results = cross_val_score(model, x, y, cv=kf, scoring='accuracy')

# Print the results
print(f"Cross-validation accuracies: {cv_results}")
print(f"Mean accuracy: {cv_results.mean()}")
print(f"Standard deviation of accuracy: {cv_results.std()}")

# Detailed results with manual loop (optional)
accuracies = []
for train_index, test_index in kf.split(x):
    
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    accuracies.append(accuracy)
    print(f"Fold accuracy: {accuracy}")

print(f"Mean accuracy: {np.mean(accuracies)}")
print(f"Standard deviation of accuracy: {np.std(accuracies)}")


# In[71]:


#Serialize the model and save
get_ipython().system('pip install joblib')
import joblib
joblib.dump(RFC, 'randomfs.pkl')
print("Random Forest Model Saved")
#Load the model
RFC = joblib.load('randomfs.pkl')
# Save features from training
rnd_columns = list(x_train.columns)
joblib.dump(rnd_columns, 'rnd_columns.pkl')
print("Random Forest Model Colums Saved")

