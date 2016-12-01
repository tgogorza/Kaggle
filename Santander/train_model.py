from clean_data import clean_data
import sklearn

df_clean = clean_data(df)
#Split columns into train data and labels
data = df_clean.ix[:,:12]       
labels = df_clean.ix[:,12:]

#Split into trainig, validation and test sets
trainds