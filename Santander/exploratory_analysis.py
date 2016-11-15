import pandas as pd
import random

#data = pd.read_csv('data/train_ver2.csv')

#let's get a smaller dataset
#sample = data.sample(frac=0.1, replace=False)

# The data to load
file = 'data/train_ver2.csv'
# Count the lines
num_lines = sum(1 for l in open(file))
# Sample size - in this case ~10%
size = int(num_lines / 10)
# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines), num_lines - size)
# Read the data
data = pd.read_csv(file, skiprows=skip_idx)
# Save the partial sample as CSV
data.to_csv('data/sample.csv', sep=",", index=False)

#Import samples
data = pd.read_csv('data/sample.csv')
