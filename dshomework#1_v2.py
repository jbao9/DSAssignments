#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:30:47 2020

@author: Jun Bao

Data Science Homework #1

Chapter 2
"""

# 11. In the botton-right window, that is the console, where we can see the 
# result after we run a line/lines of code.
# The left (for Python) window is displaying Python code we are going to run.

# 12. This is a comment!

# 13. The "run" button is on the top left of the window, the keyboard shortcut
# is F9.

# 14. The output is displaying the comment of question 13. Because the line of 
# code of question 13 are comments, which are not interpreted by Python.

# 15.
import pandas as pd
import numpy as np

#16.
# import the data set
# instead of wirte the dicionary in the python command, set the spyder woking dictionary (Files tab on the upper left ) to where we store our data set.
bank_train = pd.read_csv("bank_marketing_training")

# inspect the dateset we just imported
bank_train.head()

#17. create a contingency table
pd.crosstab(bank_train["response"], bank_train["previous_outcome"])

#18. save previous output into a varialble crosstab_01
crosstab_01 = pd.crosstab(bank_train["response"], bank_train["previous_outcome"])

#19. print the crosstab_01
crosstab_01

#20. rename the contingency table
bao2020 = crosstab_01

# inspect the dateset after saving to a differnt name
bao2020

#21. save the first nine records of the bank_train data set
bank_train_first_9 = bank_train[0:9]
bank_train_first_9

#22.
# print all the column names of bank_train dataset to inspect the data set
bank_train.columns
# save the age and marital records of the bank_train data set as their own data fram
bank_train[["age", "marital"]]

#23. save the first three records of the age and marital variables as their own data frame
bank_train_first_3_records = bank_train[["age", "marital"]].loc[[0, 1, 2]]
# or:
bank_train_first_3_records = bank_train[["age", "marital"]].loc[0: 2]
bank_train_first_3_records

#24.
# import the data set
# instead of wirte the dicionary in the python command, set the spyder woking dictionary (Files tab on the upper left ) to where we store our data set.
adult_train = pd.read_csv("adult_ch3_training")
# inspect the data set we just imported
adult_train

#25.
# rename the data set into a new name adult
adult = adult_train
# inspect the data set after changing name
adult

#26. import the library
from sklearn.tree import DecisionTreeClassifier

#27. create a contingency table of workclass and sex
table01 = pd.crosstab(adult["workclass"], adult["sex"])
# inspect the contengency table we just created
table01

#28.
# display all the column names in the data set first
adult.columns
# create the contingency table table02
table02 = pd.crosstab(adult["sex"], adult["marital-status"])
# inspect the contengency table we just created
table02

#29.
# set the console to display 10 columns, so we can see all the output
pd.set_option('display.max_columns',10)
# display the sex and workclass values of the person in the console oupput
adult[['sex', 'workclass']].iloc[0]
adult.loc[0][['sex', 'workclass']]
adult.iloc[0]
# the first record belong to Self-emp-not-inc & male category, which is in cell
# [6, 1] of table01
# there are 991 other records in the data set have the same values

#30. display the sex and marital status values of the people in records 6–10
adult[['sex', 'marital-status']].iloc[6:11]
# The result belong to category of Male vs. Married-civ-spouse and Male vs. Divorced
# which are in the cell [1, 2] and [1, 0] of table02.
# There are 6006 (Male vs. Married-civ-spouse) and 794 (Male vs. Divorced)
# other records in the data set have the same values.

#31. create a new data set that has only records whose marital status is “Married‐civ‐spouse”
adultMarried = adult[adult['marital-status'] == 'Married-civ-spouse']
# insepct the output
adultMarried

#32. recreate the contingency table of sex and workclass using the adultMarried data set
pd.crosstab(adultMarried['sex'], adultMarried['workclass'])
# Thre are much more male than female with 'Married-civ-spouse' status

#33. create a new data set that has only records whose age value is greater than 40
adultOver40 = adult[adult['age'] > 40]
# insepct the output
adultOver40

#34. recreate the contingency table of sex and marital status using the adultOver40 data set
pd.crosstab(adultOver40['sex'], adultOver40['marital-status'])
# 1) There is no people in "Married-AF-spouse" status.
# 2) The number of people in "Widowed" didn't change much comparing to table02.
# 3) The number of people in "Never-married" reduced significantly comparing to table02.


"""
@author: Jun Bao

Chapter 3
"""

from scipy import stats

#11.
# set the console to display 24 columns 
pd.set_option('display.max_columns',24)
# find how many records and columns in the data set first
bank_train.shape
# add the index column to the data set
bank_train['index'] = pd.Series(range(0,26874))
# inspect the output
bank_train

#12.
bank_train['days_since_previous'].replace({999: np.NaN})

#13.
# create a dictionary to prepare to conver categorical values to numeric values in education field
dict_edu = {'education_numeric': {'illiterate': 0, 'basic.4y': 4, 'basic.6y': 6, 'basic.9y': 9, 'high.school': 12, 'professional.course': 12, 'university.degree': 16, 'unknown': np.NaN}}
# careate a new column and replicate the value from education column, and then replace the value
bank_train['education_numeric'] = bank_train['education'].replace(dict_edu, inplace=True)
# inspect the output
bank_train

#14.
# create a new column age_z with standarized value of age column
bank_train['age_z'] = stats.zscore(bank_train['age'])
# inspect the output
bank_train
# print out the first 10 records for column age and age_z
bank_train[['age', 'age_z']].head(n=10)

#15.
# create a outlier table
bank_train_outliers = bank_train.query('age_z > 3 | age_z < -3')
# inspect the outcom
bank_train_outliers

# sort column age_z in decending order
bank_train_sort = bank_train.sort_values(['age_z'], ascending=False)
# inpect the output
bank_train_sort
# print out the list of the first 10 largest age_z
bank_train_sort['age_z'].head(10)

#16.
# visulize the job distribution in the data set to get sense of the data
bank_train['job'].value_counts().plot(kind='bar')
# calculate the percentage of different jobs in job column
job_categories_perc = bank_train['job'].value_counts(normalize=True)
# convert into dictionary format
dict_job_categories_perc = job_categories_perc.to_dict()
dict_job_categories_perc
# create a new column other and map the correlated percentage value based on the value in column job
bank_train['other'] = bank_train['job'].map(dict_job_categories_perc)
bank_train
# apply np.where() to apply 'if then condition' to the other column
bank_train['other'] = np.where(bank_train['other'] < 0.05, 'other', bank_train['job'])
bank_train


# 17.
bank_train.rename(columns = {'default': 'credit_default'}, inplace=True)
bank_train

#18.
# pinrt out the distinct value in month column first
bank_train['month'].unique()
# create a dictionary for every month 
dict_month = {'month': {'jan': 1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}}
# reaplce the value in month column to 1-12
bank_train.replace(dict_month, inplace=True)
bank_train

#19a.
bank_train['duration_z'] = stats.zscore(bank_train['duration'])
bank_train

#19b.
# calculate all the outliers and crate a new data frame duration_outliers
duration_outliers = bank_train.query('duration_z > 3 | duration_z < -3')
# inspect the outliers with only viewing duration and duration_z column
duration_outliers[['duration', 'duration_z']]
# count all the outliers
outliers_num = duration_outliers['duration_z'].count()
outliers_num
# sort duration outliers in a descending order
bank_train_duration_sort = bank_train.sort_values(['duration_z'], ascending=False)
bank_train_duration_sort
# identify the record of maximum outlier based on duration
duration_outlier_max = bank_train.loc[bank_train['duration_z'] == (bank_train['duration_z'].max())]
duration_outlier_max
# identify the record of minimum outlier
duration_outlier_min = bank_train.loc[bank_train['duration_z'] == (bank_train['duration_z'].min())]
duration_outlier_min

#20a.
bank_train['campaign_z'] = stats.zscore(bank_train['campaign'])
bank_train

#20b.
# create a new data frame campaign_outliers to list all the outliers
campaign_outliers = bank_train.query('campaign_z >3 | campaign_z < -3')
campaign_outliers
# sort the bank_train data frame based on campaign_z value in a decending order
bank_train_outliers_sort = bank_train.sort_values(['campaign_z'], ascending=False)
bank_train_outliers_sort
# identify the record of maximum outlier based on campain
campaign_outlier_max = bank_train.loc[bank_train['campaign_z'] == bank_train['campaign_z'].max()]
campaign_outlier_max
# identify the record of minum outlier based on campain
campaign_outlier_min = bank_train.loc[bank_train['campaign_z'] == bank_train['campaign_z'].min()]
campaign_outlier_min

#21a.
# import the data set
# instead of wirte the dicionary in the python command, set the spyder woking dictionary (Files tab on the upper left ) to where we store our data set.
nutrition = pd.read_csv('Nutrition_subset')
nutrition
# sort the data frame by saturated_fat_sort in a decending order and select the top 5 record
saturated_fat_sort_top5 = nutrition.sort_values('saturated_fat', ascending=False).iloc[:5]
saturated_fat_sort_top5

#21b.
# The size description is invalid for the food item. Because the unit of the food item
# are different, some are described as OZ and some are described as ENVELP,etc. These are quite
# different units, which could not be compared together.

#22a.
# creat a new column saturated_fat_per_gram
nutrition['saturated_fat_per_gram'] = nutrition['saturated_fat'] / nutrition['weight_in_grams']
nutrition
# sort the saturated_fat_per_gram in a descending order and select the top 5 records
saturated_fat_per_gram_sort = nutrition.sort_values('saturated_fat_per_gram', ascending=False).iloc[:5]
saturated_fat_per_gram_sort

#22b.
# "BUTTER; SALTED 1 TBSP" has the highest saturated fat per gram.

#23a.
# create a new column cholesterol_per_gram
nutrition['cholesterol_per_gram'] = nutrition['cholesterol'] / nutrition['weight_in_grams']
nutrition['cholesterol_per_gram']
# sort the data frame by cholesterol_per_gram_sort vaule, and select the first 5 records
cholesterol_per_gram_sort_top5 = nutrition.sort_values('cholesterol_per_gram', ascending=False).iloc[:5]
cholesterol_per_gram_sort_top5

#23b.
# "EGGS; RAW; YOLK 1 YOLK" has th most cholesterol fat per gram.

#24.
# create a new column display the zsocre for saturated_fat_per_gram
nutrition['saturated_fat_per_gram_z'] = stats.zscore(nutrition['saturated_fat_per_gram'])
nutrition
# create a data frame contains outliers at the high end of the scale
saturated_fat_per_gram_outliers_high_end = nutrition.query('saturated_fat_per_gram_z > 3')
saturated_fat_per_gram_outliers_high_end
# count the all the outliers
saturated_fat_per_gram_outlier_low_end_num = nutrition.query('saturated_fat_per_gram_z < -3').count()
saturated_fat_per_gram_outlier_low_end_num
# There is no food item is outliers at the low end of the scale for the filed of saturated_fat_per_gram

#25.
# create a new column cholesterol_per_gram_z
nutrition['cholesterol_per_gram_z'] = stats.zscore(nutrition['cholesterol_per_gram'])
nutrition
# create a data frame contains the high end outliers of cholesterol_per_gram_z_outliers_high_end_food_list
cholesterol_per_gram_z_outliers_high_end = nutrition.query('cholesterol_per_gram_z > 3')
cholesterol_per_gram_z_outliers_high_end
# create a list of food item with the high end outliers of cholesterol_per_gram_z_outliers
cholesterol_per_gram_z_outliers_high_end_food_list = cholesterol_per_gram_z_outliers_high_end['food item']
cholesterol_per_gram_z_outliers_high_end_food_list

#26.
# import file adult_ch3_training
# instead of wirte the dicionary in the python command, set the spyder woking dictionary (Files tab on the upper left ) to where we store our data set.
adult_ch3_train = pd.read_csv('adult_ch3_training')
adult_ch3_train
# Add a record index clumn to the data set
adult_ch3_train['index'] = pd.Series(range(0,14797))
adult_ch3_train

#27.
# create a new column education_z 
adult_ch3_train['education_z'] = stats.zscore(adult_ch3_train['education'])
# query out all the education outliers and save to new variable name education_outliers
education_outliers = adult_ch3_train.query('education_z > 3 | education_z < -3')
education_outliers
# There are outliers exist for the ducation field

#28a.
# create a new column age_z as the standardize value for the age colunn 
adult_ch3_train['age_z'] = stats.zscore(adult_ch3_train['age'])
adult_ch3_train

#28b.
# cretae a data frame contains all the outliers
age_outliers = adult_ch3_train.query('age_z > 3 | age_z < -3')
age_outliers
# count the number of all the outliers
age_outlier_num = age_outliers['age_z'].count()
age_outlier_num
# There are 60 outliers

# sort the outliers 
age_outlier_sort = age_outliers.sort_values('age_z', ascending=True)
age_outlier_sort
# display the maxium outliers 
age_outliers_maximum = age_outliers.loc[age_outliers['age_z'] == age_outliers['age_z'].max()]
age_outliers_maximum
# the age_outliers_maximum value is 3.751354

# display the minimum outliers
age_outliers_minimum = age_outliers.loc[age_outliers['age_z'] == age_outliers['age_z'].min()]
age_outliers_minimum
# the age_outliers_maximum value is 3.020275

#29.
# create a new column capital-gain-flag as a flag, set 0 for capital gain equals zero, and 1 otherwise
# 1st way: use if condition and .loc() function
adult_ch3_train.loc[adult_ch3_train['capital-gain']!=0, 'capital-gain-flag'] = 1
adult_ch3_train
# 2nd way: use if condition and lambda 
adult_ch3_train['capital-gain-flag'] = adult_ch3_train['capital-gain'].apply(lambda x: 0 if x == 0 else 1)
adult_ch3_train

#30.
# create a data frame age_80_and_older and contain all the records with age at least 80
age_80_and_older = adult_ch3_train.loc[adult_ch3_train['age'] >= 80]
age_80_and_older
# plot a histogram to see tha age distribution
age_80_and_older['age'].plot(kind='hist')
# Yes, there is age anomaly. There are more than 20 people with the age between 88 and 90.
# But, the the age range between 80 and 88, there are much less people.
# This is not in accordance with the laws of nature.


corsstab_04_norm_dict = corsstab_04_norm['yes'].to_dict()
bank_train['job2'] = bank_train['job'].map(corsstab_04_norm_dict)
bank_train['job2'] = np.where(bank_train['job2'] < 0.1, '0<10', np.where(bank_train['job2'] >= 0.25 , '25<33', '10<25') )








