# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
#print(data.head())
#Code starts here 
data['Gender'].replace('-','Agender',inplace=True)
gender_count=data['Gender'].value_counts()
gender_count.plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('No. of counts')
#plt.bar(gender_count)
plt.show()
#print(gender_count)



# --------------
#Code starts here
alignment=data['Alignment'].value_counts()
alignment.plot(kind='bar')
plt.xlabel('Alignment')
plt.ylabel('No. of superhero counts')
plt.title('Character Alignment')
plt.show()
#print(alignment)


# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
sc_covariance=sc_df['Strength'].cov(sc_df['Combat'])
sc_strength=sc_df['Strength'].std()
sc_combat=sc_df['Combat'].std()
sc_pearson=sc_covariance/(sc_strength*sc_combat)
print("pearson's correlation coefficient between Strength & Combat is "+str(round(sc_pearson,2)))

ic_df=data[['Intelligence','Combat']]
ic_covariance=ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence=ic_df['Intelligence'].std()
ic_combat=ic_df['Combat'].std()
ic_pearson=ic_covariance/(ic_intelligence*ic_combat)
print("pearson's correlation coefficient between Intelligence & Combat is "+str(round(ic_pearson,2)))



# --------------
#Code starts here
total_high=data['Total'].quantile(0.99)
super_best=data[data['Total']>=total_high]
super_best_names=list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3)=plt.subplots(1,3,figsize=(20,10))
data['Intelligence'].plot(kind='box', stacked=True, ax=ax_1)
ax_1.set_title('Box chart with Intelligence')

data['Speed'].plot(kind='box', stacked=True, ax=ax_2)
ax_2.set_title('Stacked bar-chart with Speed ')

data['Power'].plot(kind='box', stacked=True, ax=ax_3)
ax_3.set_title('Stacked bar-chart with Power')




