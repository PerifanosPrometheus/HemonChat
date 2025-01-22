# Import packages
import pandas as pd
import pickle
import sys

sys.path.append('..')

from utils.text_utils import is_acceptable_answer

print('Loading data...')

# Load data
with open('../../data/training_data.pkl','rb') as f:
    training_data = pickle.load(f)

print('Finished')
print('Filtering training data...')

# Find answers that follow the format requirements
training_data['is_acceptable'] = training_data['answer'].apply(lambda x: is_acceptable_answer(x))

# Only keep acceptable answers
training_data = training_data[training_data['is_acceptable']]

# Drop is acceptable column
training_data = training_data.drop(['is_acceptable'],axis=1)

print('Finished')
print('Saving Data...')

# Save data
training_data.to_pickle('../../data/training_data.pkl')
training_data.to_csv('../../data/training_data.csv', index=False)

print('Finished')