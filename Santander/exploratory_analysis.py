import pandas as pd
import random
import os
import sys
#import matplotlib.pyplot as plt
from ggplot import *
# Set current dir as working dir
os.chdir('/home/tomas/Kaggle/Santander')

#data = pd.read_csv('data/train_ver2.csv')

#let's get a smaller dataset
#sample = data.sample(frac=0.1, replace=False)


#os.chdir(os.path.dirname(os.path.realpath(__file__)))
# The data to load
file = 'data/train_ver2.csv'
# Count the lines
num_lines = sum(1 for l in open(file))
# Sample size - in this case ~10%
size = int(num_lines / 100)
# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines), num_lines - size)
# Read the data
data = pd.read_csv(file, skiprows=skip_idx, low_memory=False)
# Save the partial sample as CSV
data.to_csv('data/sample.csv', sep=",", index=False)

#Import samples
df = pd.read_csv('data/sample.csv', low_memory=False)


df.dtypes

#Clean up the data set:
#1. Convert columns to numeric values
#2. Remove unwanted columns
#3. Treat NAs (impute or remove) try fancyimpute

#Convert columns to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce', downcast='integer')
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')

#Remove unwanted columns
df.drop(['fecha_dato','ncodpers','fecha_alta','indfall','nomprov','tipodom'], inplace=True, axis=1)

#Drop rows without country, age, segmento   (we could impute the values, but that could potientially introduce some bias)
df = df[df.pais_residencia.notnull()]
df = df[df.age.notnull()]
df = df[df.segmento.notnull()]
#I consider rows with age under 18 to either have a wrong age value or to not be able to aquire new products, therefore I'll remove them  
df = df[df.age >= 18]

#Impute missing values for renta?
TODO

#We can convert some columns to boolean (indresi, indext, sexo) 
df.sexo = df.sexo.map(lambda x: 0 if x is 'V' else 1)
df.indresi = df.indresi.map(lambda x: 0 if x is 'N' else 1)
df.indext = df.indext.map(lambda x: 0 if x is 'N' else 1)
#Map segmento to values
seg_dict = { '01 - TOP':1, '02 - PARTICULARES':2, '03 - UNIVERSITARIO':3}
df.segmento.replace(seg_dict, inplace=True)

#Cut age, income into groups?
TODO

#Countries other than spain represent less than 1%, so I'm just ignoring country column altogether
spa = len(df[df.pais_residencia == 'ES'])
notspa = len(df[df.pais_residencia != 'ES'])
float(notspa) / (spa+notspa)
df.drop(['pais_residencia'], inplace=True, axis=1)

#Group countries with less than 6 users to OTHER (not sure it's so useful, data is VERY skewed towards spain)
#Maybe group ALL countries that ar not Spain into Others? 
#country_freq = df['pais_residencia'].value_counts()
#country_freq[['AR']]

#df.pais_residencia = [str(country) if country_freq[str(country)] > 5 else 'OTHER' for country in df['pais_residencia']]
#df.pais_residencia = [str(country) if country is 'ES' else 'OTHER' for country in df['pais_residencia']]

##Exploratory Analysis
df.describe()
df.head(5)

df.age.value_counts()



#Some graphs and charts
df['age'].hist(bins=20)
df['age'].apply(lambda x: math.log10(x)).hist(bins=20)

ggplot(df, aes('age','renta') + geom_bar() + scale_x_discrete(breaks=[10,20,30,40,50,60,70,80,90,100])

#Plot age vs income, coloured by seniority, split by segment
ggplot(df, aes('age','renta',colour='antiguedad')) + geom_point() + ylim(0,0.2e7) + xlim(18.120)
ggplot(df, aes('age','renta',colour='antiguedad')) + geom_point() + facet_wrap('segmento') + ylim(0,0.2e7) + xlim(18.120)

#If we plot age vs seniority, we can see a clear lower bound. We'll remove the outliers to the left of that boundary  
ggplot(df, aes('age','antiguedad')) + geom_point() + geom_abline(slope=10.42,intercept=-170, color='red', thickess=2) + ylim(-10,280) + xlim(15,120)
limit = lambda x: 10.42 * x - 170
#Let's calculate how many outliers we have with a strange age-seniority ratio
agein = len(df[df.antiguedad <= df.age.apply(limit)])
ageout = len(df[df.antiguedad > df.age.apply(limit)])
float(ageout) / (agein+ageout)
#Less than 1%, we can remove this noisy data
df = df[df.antiguedad <= df.age.apply(limit)]
ggplot(df, aes('age','antiguedad')) + geom_point() + geom_abline(slope=10.42,intercept=-170, color='red', thickess=2) + ylim(-10,280) + xlim(15,120)




#Split into training data and labels
df.dtypes
df = df.ix[:,:18]       
labels = df.ix[:,18:]