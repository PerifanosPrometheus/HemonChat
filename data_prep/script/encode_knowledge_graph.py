# Import packages
import pandas as pd
import anthropic
import sys

sys.path.append('..')

from utils.glam_utils import generate_subgraphs, encode_subgraph_to_text, summarize_encodings, save_to_pickle

print('Loading and prepping data...')

# Load Tables
concept_relationship = pd.read_csv("../../data/hemonc/concept_relationship.csv") 
concept = pd.read_csv("../../data/hemonc/concept.csv")

# Ensure consistent data types for merging
concept['concept_code'] = concept['concept_code'].astype(str)
concept_relationship['concept_code_1'] = concept_relationship['concept_code_1'].astype(str)
concept_relationship['concept_code_2'] = concept_relationship['concept_code_2'].astype(str)

# Filter out invalid relationships
valid_relationships = concept_relationship[concept_relationship['invalid_reason'].isna()]

# Merge to add concept names and classes for concept_code_1
relationships = valid_relationships.merge(
    concept[['concept_code', 'concept_name', 'concept_class_id']],
    left_on='concept_code_1',
    right_on='concept_code',
    how='left'
).rename(columns={
    'concept_name': 'concept_name_1',
    'concept_class_id': 'concept_class_id_1'
}).drop(columns=['concept_code'])

# Merge to add concept names and classes for concept_code_2
relationships = relationships.merge(
    concept[['concept_code', 'concept_name', 'concept_class_id']],
    left_on='concept_code_2',
    right_on='concept_code',
    how='left'
).rename(columns={
    'concept_name': 'concept_name_2',
    'concept_class_id': 'concept_class_id_2'
}).drop(columns=['concept_code'])

# Fill missing concept names with concept codes
relationships['concept_name_2'] = relationships['concept_name_2'].combine_first(relationships['concept_code_2'])

# Select relevant columns for analysis
relationships = relationships[[
    'concept_code_1', 'vocabulary_id_1', 'concept_name_1', 'concept_class_id_1',
    'relationship_id', 'concept_code_2', 'vocabulary_id_2', 'concept_name_2', 'concept_class_id_2'
]]

# Ensure consistency in concept code data types
relationships['concept_code_1'] = relationships['concept_code_1'].astype(str)

# Exclude specific concept classes
exclude_classes = [
    'ReferenceDOI', 'PubMedCentralURL', 'Study Group', 
    'Duration', 'Author', 'Study', 'Reference'
]
relationships = relationships[~relationships['concept_class_id_1'].isin(exclude_classes)]

# Some Hemonc concepts are pointing to themselves
# This is likely due to human error of the curator of the ontology
# We will be removing such rows
index_of_rows_pointing_to_themselves = relationships[relationships['concept_code_1']==relationships['concept_code_2']].index
relationships = relationships.drop(index_of_rows_pointing_to_themselves)

# Drop duplicates
relationships = relationships.drop_duplicates()

print('Finished')
print('Generating Subgraphs...')

# Generate subgraphs
subgraphs = generate_subgraphs(relationships)

print('Finished')
print('Encoding Subgraphs in text...')

# Encode subgraphs into text
encoded_texts = encode_subgraph_to_text(subgraphs, concept)

print('Finished')
print('Summarizing Encodings...')

# Define Anthropic client
anthropic_client = anthropic.Anthropic()

# Summarize encoded texts
summarized_texts = summarize_encodings(encoded_texts, anthropic_client)

print('Finished')
print('Saving data...')

# Save summarized encodings for later q&a generation
save_to_pickle(summarized_texts,'../../data/intermediate/summarized_texts.pkl')