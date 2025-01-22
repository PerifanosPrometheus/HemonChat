# Import packages
import pandas as pd
import pickle
import sys

sys.path.append('..')

from utils.text_utils import extract_qa

print('Loading data...')

# Load file
with open('../../data/intermediate/qa.pkl','rb') as f:
    raw_data = pickle.load(f)

data = pd.DataFrame.from_dict(raw_data, orient='index', columns=['text'])
data.index.name = 'concept_code'

print('Finished')
print('Extracting Q&A...')

# Extract qa
data['qa'] = data['text'].apply(lambda x: extract_qa(x))

# Explode Q&A pairs
exploded_df = data.explode(column=['qa'])

# Remove some noise data that didn't produce qa pairs
exploded_df = exploded_df[~exploded_df['qa'].isna()]

# Generate question and answer columns
exploded_df['question'] = exploded_df['qa'].apply(lambda x: x['question'])
exploded_df['answer'] = exploded_df['qa'].apply(lambda x: x['answer'])

print('Finished')
print('Creating and Saving Training data...')

# Create training data
training_data = exploded_df[['question','answer']].reset_index(drop=True)

# Save training data
training_data.to_pickle('../../data/training_data.pkl')
training_data.to_csv('../../data/training_data.csv', index=False)

print('Finished')