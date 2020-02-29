# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(path)
#print(df.head())
p_a=len(df[df['fico']>=700])/len(df)
#print(p_a)
p_b=len(df[df['purpose']=='debt_consolidation'])/len(df)
#print(p_b)
df1=df[df['purpose']=='debt_consolidation']
#print(df1.head())
p_a_b=len(df1[(df1['purpose']=='debt_consolidation') & (df1['fico']>=700)])/len(df1)
print(p_a_b)
result=((p_a_b*p_a)/p_b)==p_a
print(result)


# --------------
# code starts here
#print(df.head())
prob_lp=len(df[df['paid.back.loan']=='Yes'])/len(df)
print(prob_lp)
prob_cs=len(df[df['credit.policy']=='Yes'])/len(df)
print(prob_cs)

new_df=df[df['paid.back.loan']=='Yes']
#print(new_df.head())

prob_pd_cs=len(new_df[(new_df['paid.back.loan']=='Yes') & (new_df['credit.policy']=='Yes')])/len(new_df)
print(prob_pd_cs)
bayes=(prob_pd_cs*prob_lp)/prob_cs
print(bayes)
# code ends here


# --------------
# code starts here
#print(df.head())
df['purpose'].value_counts().plot(kind='bar')
plt.ylabel('No of Counts')
plt.xlabel('Purpose')
plt.show()
df1=df[df['paid.back.loan']=='No']
df1['purpose'].value_counts().plot(kind='bar')
plt.ylabel('No of Counts')
plt.xlabel('Purpose')
plt.show()

# code ends here


# --------------
# code starts here
print(df.head())
inst_median=np.median(df['installment'])
print(inst_median)
inst_mean=np.mean(df['installment'])
print(inst_mean)
plt.hist(df['installment'])
plt.show()
plt.hist(df['log.annual.inc'])
plt.show()
#plt.hist()
# code ends here


