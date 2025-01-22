# Import packages
import pandas as pd
import pickle
import sys

sys.path.append('..')

from utils.text_utils import extract_eval_qa

print('Loading data...')

# Load data
with open('../../data/intermediate/eval_qa.pkl','rb') as f:
    data = pickle.load(f)

print('Finished')
print('Extracting Q&A...')

# Extract qa
data['qa'] = data['evaluation_qa'].apply(lambda x: extract_eval_qa(x))

# Explode Q&A pairs
exploded_df = data.explode(column=['qa'])

# Remove some noise data that didn't produce qa pairs
exploded_df = exploded_df[~exploded_df['qa'].isna()]

# Generate question and answer columns
exploded_df['question'] = exploded_df['qa'].apply(lambda x: x['question'])
exploded_df['answer'] = exploded_df['qa'].apply(lambda x: x['answer'])

print('Finished')
print('Creating and Saving Evaluation data...')

# Create eval data
eval_data = exploded_df[['question','answer']].reset_index(drop=True)

# Save eval data
eval_data.to_pickle('../../data/eval_data.pkl')
eval_data.to_csv('../../data/eval_data.csv', index=False)

print('Finished')