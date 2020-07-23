# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df=pd.read_csv(path)

X=df.drop(['customerID','Churn'],axis=1)

y=df['Churn'].copy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)




# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges']=X_train['TotalCharges'].replace(' ',np.nan)
X_test['TotalCharges']=X_test['TotalCharges'].replace(' ',np.nan)

X_train['TotalCharges']=X_train['TotalCharges'].astype(float)
X_test['TotalCharges']=X_test['TotalCharges'].astype(float)

mean=X_train['TotalCharges'].mean()

X_train['TotalCharges']=X_train['TotalCharges'].fillna(mean)
X_test['TotalCharges']=X_test['TotalCharges'].fillna(mean)

print(X_train.isnull().sum())

integer=list()
obj=list()
columns=X_train.columns
for i in columns:
    if(X_train[i].dtypes=='object'):
        obj.append(i)
    else:
        integer.append(i)

le=LabelEncoder()

for i in obj:
    X_train[i]=le.fit_transform(X_train[i])
    X_test[i]=le.transform(X_test[i])


y_train=y_train.replace({'No':0, 'Yes':1})
y_test=y_test.replace({'No':0, 'Yes':1})


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

ada_model=AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred=ada_model.predict(X_test)

ada_score=accuracy_score(y_test,y_pred)
print("Adaboost score is ",ada_score)

ada_cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix is ",ada_cm)

ada_cr=classification_report(y_test,y_pred)
print("Classification report is ",ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model=XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)

xgb_score=accuracy_score(y_test,y_pred)
print("Accuracy score is ",xgb_score)

xgb_cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix score is ",xgb_cm)

xgb_cr=classification_report(y_test,y_pred)
print("classification report is ",xgb_cr)


clf_model=GridSearchCV(estimator=xgb_model,param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred=clf_model.predict(X_test)

clf_score=accuracy_score(y_test,y_pred)
print("Accuracy score is ",clf_score)

clf_cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix score is ",clf_cm)

clf_cr=classification_report(y_test,y_pred)
print("classification report is ",clf_cr)





