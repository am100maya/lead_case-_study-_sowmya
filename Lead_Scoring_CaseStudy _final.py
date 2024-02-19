#!/usr/bin/env python
# coding: utf-8

# ---
# # Lead Scoring Case Study
# 
# Goals of the Case Study:
# 
# 1. Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.
# 
# 2. There are some more problems presented by the company which your model should be able to adjust to if the company's requirement changes in the future so you will need to handle these as well.
# 
# ---
# 

# ## Adding Required Libraries as per requirement

# In[104]:


import warnings
warnings.filterwarnings('ignore')

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# visulaisation
from matplotlib.pyplot import xticks
get_ipython().run_line_magic('matplotlib', 'inline')

# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# ## Data Preparation

# ### Data Loading 

# In[105]:


data = pd.read_csv(r"C:\Users\am100\Downloads\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")
data.head(5)


# ###Checking duplicates

# In[106]:


sum(data.duplicated(subset = 'Prospect ID')) == 0
# No duplicate values


# In[107]:


data.shape


# In[108]:


data.info()


# In[109]:


data.describe()


# ---

# ## Data Cleaning

# As we can observe that there are select values for many column.
# This is because customer did not select any option from the list, hence it shows select.
# Select values are as good as NULL.

# In[110]:


# Converting 'Select' values to NaN.
data = data.replace('Select', np.nan)


# In[111]:


data.isnull().sum()


# In[112]:


round(100*(data.isnull().sum()/len(data.index)), 2)


# In[113]:


# we will drop the columns having more than 70% NA values.
data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, axis = 1)
data.shape


# **Lead Quality column indicates the quality of lead based on the data.**

# In[114]:


data['Lead Quality'].describe()


# In[115]:


sns.countplot( data= data, x = data['Lead Quality'])


# In[116]:


# As Lead quality is based on the intution of employee, so if left blank we can impute 'Not Sure' in NaN safely.
data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')


# In[117]:


sns.countplot( data= data, x = data['Lead Quality'])


# **There is too much variation in thes parameters so its not reliable to impute any value in it.** 
# **45% null values means we need to drop these columns.**

# In[118]:


data.info()


# In[119]:


round(100*(data.isnull().sum()/len(data.index)), 2)


# ###City

# In[120]:


data.City.describe()


# In[121]:


data["City"]=data["City"].replace(  np.nan,'Mumbai')


# **Around 60% of the data is Mumbai so we can impute Mumbai in the missing values.** 

# In[122]:


data['City'] = data['City'].replace(np.nan, 'Mumbai')


# ### Specailization

# In[123]:


data.Specialization.describe()


# In[124]:


sns.countplot(data = data, x=data.Specialization)
xticks(rotation = 90)


# **It maybe the case that lead has not entered any specialization if his/her option is not availabe on the list, may not have any specialization or is a student.**
# **Hence we can make a category "Others" for missing values.**

# In[125]:


data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')


# In[126]:


round(100*(data.isnull().sum()/len(data.index)), 2)


# ###Tags

# In[127]:


data.Tags.describe()


# In[128]:


fig, axs = plt.subplots(figsize = (15,7.5))
sns.countplot(data = data, x=data.Tags)
xticks(rotation = 90)


# In[129]:


# Blanks in the tag column may be imputed by 'Will revert after reading the email'.
data['Tags'] = data['Tags'].replace(np.nan, 'Will revert after reading the email')


# ### What matters most to you in choosing a course 

# In[130]:


data['What matters most to you in choosing a course'].describe()


# In[131]:


# Blanks in the this column may be imputed by 'Better Career Prospects'.
data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')


# ### Occupation

# In[132]:


data['What is your current occupation'].describe()


# **86% entries are of Unemployed so we can impute "Unemployed" in it.** 

# ### What is your current occupation 

# In[133]:


data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')


# ### Country

# In[134]:


# Country is India for most values so let's impute the same in missing values.
data['Country'] = data['Country'].replace(np.nan, 'India')


# In[135]:


round(100*(data.isnull().sum()/len(data.index)), 2)


# In[136]:


# Rest missing values are under 2% so we can drop these rows.
data.dropna(inplace = True)


# In[137]:


round(100*(data.isnull().sum()/len(data.index)), 2)


# **The data is clean and can be taken for EDA**

# # Exploratory Data Analysis for Case Study

# ## Univariate Analysis

# ### Converted

# In[138]:


# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0).
Converted = (sum(data['Converted'])/len(data['Converted'].index))*100
Converted


# ### Lead origin

# In[139]:


sns.countplot(x = "Lead Origin", hue = "Converted", data = data)
xticks(rotation = 90)


# #### **Inference**
# 
# API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# 
# Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# 
# Lead Import are very less in count.
# 
# **To improve overall lead conversion rate, the focus will be on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.**

# ### Lead Source

# In[140]:


fig, axs = plt.subplots(figsize = (15,7.5))
sns.countplot(x = "Lead Source", hue = "Converted", data = data)
xticks(rotation = 90)


# In[141]:


data['Lead Source'] = data['Lead Source'].replace(['google'], 'Google')
data['Lead Source'] = data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')


# In[142]:


fig, axs = plt.subplots(figsize = (15,7))
sns.countplot(x = "Lead Source", hue = "Converted", data = data)
xticks(rotation = 30)


# #### Inference
# 1. Google and Direct traffic generates maximum number of leads.
# 2. Conversion Rate of reference leads and leads through welingak website is high.
# 
# #### To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# ### Do Not Email & Do Not Call

# In[143]:


fig, axs = plt.subplots(1,2,figsize = (15,7.5))
sns.countplot(x = "Do Not Email", hue = "Converted", data = data, ax = axs[0])
sns.countplot(x = "Do Not Call", hue = "Converted", data = data, ax = axs[1])


# ### Total Visits

# In[144]:


data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[145]:


sns.boxplot(data['TotalVisits'])


# ####Inference:
# 
# As we can see there are a number of outliers in the data.
# We will cap the outliers to 95% value for analysis.

# In[146]:


percentiles = data['TotalVisits'].quantile([0.05,0.95]).values
data['TotalVisits'][data['TotalVisits'] <= percentiles[0]] = percentiles[0]
data['TotalVisits'][data['TotalVisits'] >= percentiles[1]] = percentiles[1]


# In[147]:


sns.boxplot(data['TotalVisits'])


# In[148]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = data)


# **Median for converted and non-converted leads are the same , hence Total visits are of no importance for further analysis.** 

# ### Total time spent on website
# 

# In[149]:


data['Total Time Spent on Website'].describe()


# In[150]:


sns.boxplot(data['Total Time Spent on Website'])


# In[151]:


sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = data)


# In[152]:


sns.barplot(y = 'Total Time Spent on Website', x = 'Converted', data = data)


# #### Inference
# 1. Leads spending more time on the weblise are more likely to be converted.
# 
# **Website should be made more engaging to make leads spend more time.** 

# ### Page views per visit

# In[153]:


data['Page Views Per Visit'].describe()


# In[154]:


sns.boxplot(data['Page Views Per Visit'])


# **As we can see there are a number of outliers in the data.**
# **We will cap the outliers to 95% value for analysis.**

# In[155]:


percentiles = data['Page Views Per Visit'].quantile([0.05,0.95]).values
data['Page Views Per Visit'][data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
data['Page Views Per Visit'][data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]


# In[156]:


sns.boxplot(data['Page Views Per Visit'])


# In[157]:


sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = data)


# **Median seems equal for converted and non converted, hence nothing conclusive can be said for Page Views Per Visit**

# ### Last Activity

# In[158]:


data['Last Activity'].describe()


# In[159]:


fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(x = "Last Activity", hue = "Converted", data = data)
xticks(rotation = 60)


# In[160]:


# Let's keep considerable last activities as such and club all others to "Other_Activity"
data['Last Activity'] = data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')


# In[161]:


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Last Activity", hue = "Converted", data = data)
xticks(rotation = 60)


# **1. Most of the lead have their Email opened as their last activity.**
# 
# **2. Conversion rate for leads with last activity as SMS Sent is almost 60%.**

# ### Country

# In[162]:


data.Country.describe()


# #### Inference:
# Most values are 'India' no such inference can be drawn

# ### Specialization

# In[163]:


data.Specialization.describe()


# In[164]:


data['Specialization'] = data['Specialization'].replace(['Others'], 'Other_Specialization')


# In[165]:


fig, axs = plt.subplots(figsize = (18,8))
sns.countplot(x = "Specialization", hue = "Converted", data = data)
xticks(rotation = 75)


# #### Inference
# 1. Focus should be more on the Specialization with high conversion rate.
# 
# **We can focus here on Management related Professions.**

# ### Occupation

# In[166]:


data['What is your current occupation'].describe()


# In[167]:


fig, axs = plt.subplots(figsize = (20,8))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = data)
xticks(rotation = 90)


# #### Inference
# 1. Working Professionals going for the course have high chances of joining it.
# 2. Unemployed leads are the most in numbers but has around 30-35% conversion rate.

# ### What matters most to you in choosing a course

# In[168]:


data['What matters most to you in choosing a course'].describe()


# **Most entries are 'Better Career Prospects'. No Inference can be drawn with this parameter**

# ### Search

# In[169]:


data.Search.describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Magazine
# 

# In[170]:


data.Magazine.describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Newspaper Article
# 

# In[171]:


data['Newspaper Article'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### X Education Forums

# In[172]:


data['X Education Forums'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Newspaper

# In[173]:


data['Newspaper'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Digital Advertisement

# In[174]:


data['Digital Advertisement'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Through Recommendations¶
# 

# In[175]:


data['Through Recommendations'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Receive More Updates About Our Courses

# In[176]:


data['Receive More Updates About Our Courses'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Tags
# 

# In[177]:


data.Tags.describe()


# In[178]:


fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(x = "Tags", hue = "Converted", data = data)
xticks(rotation = 90)


# In[179]:


# Let's keep considerable last activities as such and club all others to "Other_Activity"
data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')


# In[180]:


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Tags", hue = "Converted", data = data)
xticks(rotation = 90)


# ### Lead Quality

# In[181]:


data['Lead Quality'].describe()


# In[182]:


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Lead Quality", hue = "Converted", data = data)
xticks(rotation = 90)


# ### Update me on Supply Chain Content
# 

# In[183]:


data['Update me on Supply Chain Content'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### Get updates on DM Content

# In[184]:


data['Get updates on DM Content'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.** 

# ### I agree to pay the amount through cheque

# In[185]:


data['I agree to pay the amount through cheque'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### A free copy of Mastering The Interview

# In[186]:


data['A free copy of Mastering The Interview'].describe()


# **Most entries are 'No'. No Inference can be drawn with this parameter.**

# ### City
# 

# In[187]:


data.City.describe()


# In[188]:


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "City", hue = "Converted", data = data)
xticks(rotation = 90)


# **Most leads are from mumbai with around 30% conversion rate.**

# ### Last Notable Activity

# In[189]:


data['Last Notable Activity'].describe()


# In[190]:


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Last Notable Activity", hue = "Converted", data = data)
xticks(rotation = 90)


# **Based on Univariate analysis, many columns are not adding any values to the method. So it is better drop these columns.**

# In[191]:


data = data.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country', 'Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score','Do Not Call','Do Not Email'],axis =1)


# In[192]:


data.shape


# In[193]:


data.head()


# ---

# # Data Preparation

# #### For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[194]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)
dummy1.head()


# In[195]:


# Adding the results to the master dataframe
data = pd.concat([data, dummy1], axis=1)
data.head()


# In[196]:


data = data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'], axis = 1)


# In[197]:


data.head()


# In[198]:


from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = data.drop(['Prospect ID','Converted'], axis=1)


# In[199]:


X.head()


# In[200]:


# Putting response variable to y
y = data['Converted']

y.head()


# In[201]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ## Feature Scaling
# 

# In[202]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# # Model Building

# ## Running Your First Training Model

# In[203]:


import statsmodels.api as sm


# In[205]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train.astype(float))), family = sm.families.Binomial())
logm1.fit().summary()


# ## Feature Selection Using RFE
# 

# In[210]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE 
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=15)
#rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[211]:


rfe.support_


# In[212]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[213]:


col = X_train.columns[rfe.support_]
col


# In[214]:


X_train.columns[~rfe.support_]


# ## Assessing the model with StatsModels

# In[217]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm.astype(float), family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[218]:


col1 = col.drop('Tags_wrong number given',1)


# In[220]:


X_train_sm = sm.add_constant(X_train[col1])
logm2 = sm.GLM(y_train,X_train_sm.astype(float), family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[221]:


col2 = col1.drop('Tags_Not doing further education',1)


# In[223]:


X_train_sm = sm.add_constant(X_train[col2])
logm2 = sm.GLM(y_train,X_train_sm.astype(float), family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[224]:


col3 = col2.drop('Tags_Closed by Horizzon',1)


# In[226]:


X_train_sm = sm.add_constant(X_train[col3])
logm2 = sm.GLM(y_train,X_train_sm.astype(float), family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[228]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm.astype(float))
y_train_pred[:10]


# In[229]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[230]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[231]:


y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[232]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[233]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# ## Checking VIF

# In[234]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[235]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col2].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col2].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## Metrics beyond simply accuracy
# 
# 

# In[237]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[238]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[239]:


# Let us calculate specificity
TN / float(TN+FP)


# In[240]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[241]:


# positive predictive value 
print (TP / float(TP+FP))


# In[242]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Plotting the ROC Curve

# #### An ROC curve demonstrates several things:
# 
# 1. It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# 2. The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# 3. The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[243]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[245]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[246]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# ## Finding Optimal Cutoff Point
# 

# In[247]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[248]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[249]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# ### Observation:
# 
# From the curve above, 0.25 is the optimum point to take it as a cutoff probability.

# In[250]:


# From the curve above, 0.25 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.25 else 0)

y_train_pred_final.head()


# ## Assigning Lead Score

# In[251]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final.head()


# In[252]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[253]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[254]:


# Let us calculate specificity
TN / float(TN+FP)


# In[255]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[256]:


# Negative predictive value
print (TN / float(TN+ FN))


# # Precision and Recall
# 

# In[257]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion


# In[258]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[259]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[260]:


from sklearn.metrics import precision_score, recall_score


# In[261]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted)


# In[262]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)


# ## Precision and Recall Tradeoff

# In[263]:


from sklearn.metrics import precision_recall_curve


# In[264]:


y_train_pred_final.Converted, y_train_pred_final.predicted


# In[265]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[266]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ## Making predictions on the test set
# 

# In[267]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[268]:


X_test = X_test[col3]
X_test.head()


# In[273]:


X_test_sm = sm.add_constant(X_test)


# In[275]:


y_test_pred = res.predict(X_test_sm.astype(float))


# In[276]:


y_test_pred[:10]


# In[278]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[279]:


# Let's see the head
y_pred_1.head()


# In[280]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[281]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[282]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[283]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[284]:


y_pred_final.head()


# In[285]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[286]:


# Rearranging the columns
columns_titles = ['Prospect ID','Converted','Converted_prob']
y_pred_final=y_pred_final.reindex(columns=columns_titles)


# In[287]:


y_pred_final.head()


# In[288]:


y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.25 else 0)


# In[289]:


y_pred_final.head()


# # Let's check the overall Accuracy.

# In[290]:


metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)


# In[291]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2


# In[292]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# # Let's Calculate Sensitivity of our Logistic Regression Model

# In[293]:


TP / float(TP+FN)


# # Let's Calculate Specificity of Model

# In[294]:


TN / float(TN+FP)


# # Observations:
# 
# ##After running the model on the Train Dataset these are the figures we obtain:
# 
# * Accuracy : 89.98%
# * Sensitivity : 81.80%
# * Specificity : 94.96%
# 
# ##After running the model on the Test Dataset these are the figures we obtain:
# 
# * Accuracy : 83.37%
# * Sensitivity : 85.52%
# * Specificity : 91.58%

# In[ ]:





# In[ ]:





# In[ ]:




