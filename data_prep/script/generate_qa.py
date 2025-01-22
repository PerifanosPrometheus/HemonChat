# Import Packages
import pickle
import anthropic
import sys

sys.path.append('..')

from utils.glam_utils import save_to_pickle, generate_qa

print('Loading and prepping data...')

# Load data
with open('../../data/intermediate/summarized_texts.pkl','rb') as f:
    summarized_texts = pickle.load(f)

print('Finished')
print('Generating Q&A...')

# Define Anthropic client
anthropic_client = anthropic.Anthropic()

# Define Q&A generation prompt
qa_generation_prompt = '''Task Description: The provided context contains information about oncology related drugs, procedures, etc. Your task is to thoroughly analyze the text and generate comprehensive Q&A pairs that capture ALL the key information provided. Consider the following aspects, but create questions for ANY important information present in the text:

- Drug classification and characteristics
- Mechanism of action and targets
- Regulatory approvals and timelines
- Clinical indications and approved uses
- Administration details
- Historical significance and development

Format Requirements:
Q1: [Comprehensive question that allows for detailed answer]
A1: [Complete answer synthesizing all relevant information from the text]

Important guidelines:
- Extract ALL relevant information from the text - don't miss any key details
- Create questions that allow for comprehensive answers rather than single-fact responses
- Combine related information into coherent Q&A pairs
- Answers should synthesize information while staying true to the source text
- If multiple related pieces of information exist, combine them into one Q&A pair'''

# Generate Q&A
qa = generate_qa(summarized_texts,anthropic_client,qa_generation_prompt)

print('Finished')
print('Saving Results...')

# Save Results
save_to_pickle(qa,'../../data/intermediate/qa.pkl')

print('Finished')