# Import Packages
import pandas as pd
import pickle
import random
import anthropic
import sys

sys.path.append('..')

from utils.glam_utils import save_to_pickle, generate_eval_qa

print('Loading and prepping data...')

# Load data
with open('../../data/intermediate/qa.pkl','rb') as f:
    qa = pickle.load(f)

# Store qa in a dataframe
qa = pd.DataFrame.from_dict(qa, orient='index', columns=['qa'])
qa.index.name = 'concept_code'

print('Finished')

# Define evaluation dataset size
n_eval = int(len(qa)*0.05)

print(f'Subsetting data for evaluation using size: {n_eval}...')

# Define list of potential indeces
all_idx = list(qa.index)

# Subset data
eval_idx = []
for _ in range(n_eval):
    found_valid_idx = False
    while not found_valid_idx:
        # choose random index
        idx = random.choice(all_idx)
        # Remove index from all possible indices to choose from
        all_idx.remove(idx)
        # check if qa text does not contain words such as context or text or information which would make the entry unaccapetable
        qa_text = qa.loc[idx,'qa']
        if ('context' not in qa_text) and ('text' not in qa_text) and ('information' not in qa_text):
            # Append index to the eval list of indices
            eval_idx.append(idx)
            # Stop iterating
            found_valid_idx = True
        else:
            # Try finding another index
            found_valid_idx = False

# Filter data to eval indices and rename column
eval_qa = qa[qa.index.isin(eval_idx)]
eval_qa = eval_qa.rename({'qa':'training_qa'}, axis=1)

print('Finished')
print('Generating Evaluation Q&A...')

# Define Anthropic client
anthropic_client = anthropic.Anthropic()

# Define Q&A generation prompt
eval_generation_prompt = '''Task Description: You will be provided Q&A pairs which are part of a training dataset for LLM finetuning. Generate 1 specific question and its corresponding answer that can be used for training evaluation of the Q&A pairs provided.

You MUST generate the Q&A pair in the following format:
Q1: [Question]
A1: [Answer that can be directly verified from the text]

Important guidelines:
- If there are multiple training question and answers, try to create a evaluation question and answer pair that combines them as much as possible
- Keep the question specific and unambiguous
- Avoid yes/no questions
- Remain faithful to the original content while varying verbal form or structure
'''

# Generate evaluation Q&A pairs
# TODO: Modify the following so that it produces a log of each node it's processing.
eval_qa['evaluation_qa'] = eval_qa['training_qa'].apply(lambda x: generate_eval_qa(x,anthropic_client,eval_generation_prompt,save_progress=False))

# Drop training_qa
eval_qa = eval_qa.drop(['training_qa'],axis=1)

print('Finished')
print('Saving Results...')

# Save Results
save_to_pickle(eval_qa,'../../data/intermediate/eval_qa.pkl')

print('Finished')