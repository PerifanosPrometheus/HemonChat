import pandas as pd
from collections import defaultdict
import pickle
import os

intermediate_data_folder = '/Users/giorgiodisalvo/repos/Hemonc/HemonChat/data/intermediate'

def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - file_path: The path to the pickle file (e.g., '*.pkl').
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")

def generate_subgraphs(relationships, k=2, N_max=100, target_nodes=None):
    """
    Generate subgraphs with a neighborhood of k hops and limit the size to N_max nodes.
    
    Parameters:
    - relationships: pd.DataFrame, the relationships table.
    - k: int, number of hops to consider.
    - N_max: int, maximum number of nodes in the subgraph.
    
    Returns:
    - subgraphs: dict, mapping each node to its subgraph.
    """
    # Build adjacency list for the graph
    adjacency_list = defaultdict(set)
    for _, row in relationships.iterrows():
        adjacency_list[row['concept_code_1']].add((row['concept_code_2'], row['relationship_id']))

    # If no target_nodes provided, use all unique nodes in the table
    if target_nodes is None:
        target_nodes = relationships['concept_code_1'].unique()

    # Generate subgraphs
    subgraphs = {}
    
    for node in target_nodes:
        if node not in adjacency_list:
            continue  # Skip if the node is not in the adjacency list

        # Perform BFS to gather neighbors up to k hops
        visited = set()
        queue = [(node, 0)]  # (current_node, current_depth)
        subgraph_edges = []
        
        while queue:
            current_node, depth = queue.pop(0)
            if depth > k or current_node in visited:
                continue
            
            visited.add(current_node)
            for neighbor, relation in adjacency_list[current_node]:
                subgraph_edges.append((current_node, relation, neighbor))
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        # Limit subgraph size to N_max nodes
        unique_nodes = {edge[2] for edge in subgraph_edges}  # Gather unique nodes (concept_code_2)
        if len(unique_nodes) > N_max:
            subgraph_edges = subgraph_edges[:N_max]  # Truncate to fit N_max
        
        # Store subgraph as DataFrame for convenience
        subgraphs[node] = pd.DataFrame(subgraph_edges, columns=['concept_code_1', 'relationship_id', 'concept_code_2'])
    
    return subgraphs

def encode_subgraph_to_text(subgraphs, concept):
    """
    Encode subgraphs into the format shown in the GLAM paper.
    
    Parameters:
    - subgraphs: dict, mapping each node to its subgraph DataFrame.
    - concept: pd.DataFrame, the hemonc concept table.

    Returns:
    - glam_encoded_texts: dict, mapping each node to its GLAM-formatted text representation.
    """
    glam_encoded_texts = {}

    # Create a dictionary to map concept codes to their names and attributes
    concept_details_df = concept[['concept_code', 'concept_name', 'concept_class_id']]
    concept_details = concept_details_df.set_index('concept_code').to_dict('index')

    for node, subgraph in subgraphs.items():
        if node not in concept_details:
            continue  # Skip nodes not in the concept details

        # Store concept name of node
        node_concept_name = concept_details[node]['concept_name']

        # Group targets of the same relationship
        relationship_groups = {}
        for _, row in subgraph.iterrows():
            relationship = row['relationship_id']
            target_name = concept_details.get(row['concept_code_2'], {}).get('concept_name', row['concept_code_2'])

            if relationship not in relationship_groups:
                relationship_groups[relationship] = []
            relationship_groups[relationship].append(target_name)
        
        # Generate per-relationship encoding
        grouped_sentences = []
        for relationship, targets in relationship_groups.items():
            target_list = ",".join(sorted(set(targets)))  # Combine and deduplicate targets
            grouped_sentences.append(f"{node_concept_name}, [{relationship}], {target_list}.")
        
        # Combine sentences into GLAM summary
        adjacency_list_summary = " ".join(grouped_sentences)
        glam_encoded_texts[node] = f"{adjacency_list_summary}"
    
    return glam_encoded_texts

def summarize_encodings(
        encoded_texts, 
        anthropic_client, 
        system_prompt="You are a medical oncology journal editor.",
        summarization_prompt="Given a sentence you respond with a concise and accurate rewritten version. Ensure human-readable names, reduce redundancy, and include synonyms or expanded terms where appropriate.",
        save_progress=True
    ):
    """
    Summarize GLAM-formatted adjacency list encodings using an LLM.

    Parameters:
    - encoded_texts: dict, mapping each node to its GLAM-formatted adjacency list representation.
    - anthropic_client: object, anthropic client.
    - system_prompt: str, the system prompt defining the agent role.
    - summarization_prompt: str, the prompt to guide the agent for summarization.
    - save_progress: bool, boolean to save progress after every model response.

    Returns:
    - summarized_encodings: dict, mapping each node to its summarized representation.
    """
    summarized_encodings = {}

    for node, encoded_text in encoded_texts.items():
        print(f"Processing node: {node}...")
        # Construct the full prompt for the LLM
        input_text = f"{summarization_prompt}\n---\nSentence: {encoded_text}"

        try:
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307", #cheapest model. For this task we can probably also use Llama models.
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": input_text}
                ]
            )
            summarized_encodings[node] = response.content[0].text

            if save_progress:
                save_to_pickle(summarized_encodings,os.path.join(intermediate_data_folder,'summarized_encodings.pkl'))

        except Exception as e:
            print(f"Error processing node {node}: {e}")
            summarized_encodings[node] = None  # Handle errors gracefully

    return summarized_encodings

def generate_qa(
        summarized_texts, 
        anthropic_client, 
        qa_generation_prompt,
        system_prompt="You are a medical oncology journal editor.",
        save_progress=True
    ):
    """
    Generate Q&A from summarized GLAM encodings.

    Parameters:
    - summarized_texts: dict, mapping each node to its GLAM text encoding summarized.
    - anthropic_client: object, anthropic client.
    - qa_generation_prompt: str, the prompt to guide the agent for Q&A generation.
    - system_prompt: str, the system prompt defining the agent role.
    - save_progress: bool, boolean to save progress after every model response.

    Returns:
    - qa: dict, mapping each node to its Q&A generated.
    """
    qa = {}

    for node, text in summarized_texts.items():
        print(f"Processing node: {node}...")
        # Construct the full prompt for the LLM
        input_text = f"{qa_generation_prompt}\n---\nContext: {text}"

        try:
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307", #cheapest model. For this task we can probably also use Llama models.
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": input_text}
                ]
            )
            qa[node] = response.content[0].text

            if save_progress:
                save_to_pickle(qa,os.path.join(intermediate_data_folder,'qa_intermediate.pkl'))

        except Exception as e:
            print(f"Error processing node {node}: {e}")
            qa[node] = None  # Handle errors gracefully

    return qa

def generate_eval_qa(
        qa, 
        anthropic_client, 
        eval_generation_prompt,
        system_prompt="You are an expert AI trainer specialized in generating evaluation datasets based on training data.",
        save_progress=True
    ):
    """
    Generate evaluation Q&A from training Q&A pairs.

    Parameters:
    - qa: str, Q&A pairs for a given concept.
    - anthropic_client: object, anthropic client.
    - eval_generation_prompt: str, the prompt to guide the agent for evaluation Q&A generation.
    - system_prompt: str, the system prompt defining the agent role.
    - save_progress: bool, boolean to save progress after every model response.

    Returns:
    - eval_qa: str, Q&A to be used during model evaluation.
    """
    # Construct the full prompt for the LLM
    input_text = f"{eval_generation_prompt}\n---\nTraining Q&A pairs: {qa}"

    try:
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307", #cheapest model. For this task we can probably also use Llama models.
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        eval_qa = response.content[0].text
        if save_progress:
            save_to_pickle(eval_qa,os.path.join(intermediate_data_folder,'eval_qa_intermediate.pkl'))

    except Exception as e:
        print(f"Error processing: {e}")
        eval_qa = -1
    return eval_qa