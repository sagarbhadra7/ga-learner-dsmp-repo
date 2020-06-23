# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df=pd.read_csv(path)
print(df.head())
X=df.iloc[:,:-1]
y=df['insuranceclaim']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=6)


# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
q_value=np.percentile(X_train['bmi'], 95)  # Q3#X_train.bmi.quantile([0.95])
print(q_value)
plt.boxplot(X_train['bmi'])
plt.show()
print(y_train.value_counts())
# Code ends here


# --------------
# Code starts here
relation=X_train.corr()
print(relation)
sns.pairplot(X_train)


# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols=['children','sex','region','smoker']
fig,axes=plt.subplots(nrows = 2 , ncols = 2)
for i in range(2):
    for j in range(2):
        col=cols[ i * 2 + j]
        sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr=LogisticRegression(random_state=9)
grid=GridSearchCV(lr,parameters)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score=roc_auc_score(y_test,y_pred)
y_pred_proba=grid.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_proba)
roc_auc= metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = "Logistic model, auc="+str(roc_auc))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Code ends here


