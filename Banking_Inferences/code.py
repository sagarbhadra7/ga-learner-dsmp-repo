# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data=pd.read_csv(path)
n=sample_size

data_sample=data.sample(n,random_state=0)
sample_mean=round(np.mean(data_sample['installment']),2)
sample_std=round(data_sample['installment'].std(),2)

#print(sample_mean)
print(sample_std)

margin_of_error=round(z_critical*(sample_std/np.sqrt(sample_size)),2)
#print(margin_of_error)

confidence_interval=(sample_mean-margin_of_error,sample_mean+margin_of_error)
#print(confidence_interval)

true_mean=round(np.mean(data['installment']),2)
print(true_mean)
print(confidence_interval[0],confidence_interval[1])
if ((confidence_interval[0]<=true_mean) or (true_mean>=confidence_interval[1])):
    print("True mean falls between the confidence interval")
else:
    print("True mean does not falls between the confidence interval")



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig,axes=plt.subplots(nrows = 3 , ncols = 1)
for i in range(len(sample_size)):
    m=list()
    for j in range(1000):
        n=sample_size[i]
        mean=data['installment'].sample(n).mean()
        m.append(mean)
    mean_series=pd.Series(m)
    axes[i].hist(mean_series)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate']=pd.to_numeric(data['int.rate'].map(lambda x: x.rstrip('%')))
data['int.rate']=data['int.rate']/100
z_statistic,p_value=ztest(x1=data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
print(z_statistic,p_value)
if p_value<0.05:
    print(" reject the null hypothesis")
else:
    print('fail to reject null hypothesis')



# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
x1=data[data['paid.back.loan']=='No']['installment']
x2=data[data['paid.back.loan']=='Yes']['installment']
z_statistic,p_value=ztest(x1,x2)

if p_value<0.05:
    print('Reject Null Hypothesis:')
else:
    print('Fail to reject Null Hypothesis')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()

observed=pd.concat([yes.transpose(),no.transpose()],axis=1,keys=['Yes','No'])

chi2, p, dof, ex=chi2_contingency(observed)
if chi2 <critical_value:
    print("Reject the null hypothesis that the two distributions are the same")
else:
    print("Fail to reject the null hypothesis that the two distributions are the same")
#print(chi2, p, dof, ex)


