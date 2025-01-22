<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
-->
[![Contributors][contributors-shield]][contributors-url]



<!-- PROJECT INFO -->
<br />
<div align="center">
<h3 align="center">HemonChat</h3>

  <p align="center">
    From hemonc.org ontology data to a finetuned Llama-2-7b-chat-hf model and a simple chatbot application.
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#background">Background</a>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
  </ol>
</details>


<!-- Background -->
## Background

[HemOnc.org](https://hemonc.org/wiki/Main_Page) is the largest freely available medical wiki of interventions, regimens, and general information relevant to the fields of hematology and oncology. It is designed for easy use and intended for healthcare professionals.

For data professional, the hemonc team has released their [ontology](https://hemonc.org/wiki/Ontology) which is freely available for academic and non-commercial use via [HemOnc Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FPO4HB).

The ontology is structured based upon the OMOP data model and thus revolves around two main tables. One table contains concepts, this can be things such as drugs, diseases, treatment types, etc. The other contains a series of relationships between concepts. 

Here is a fictitious example that shows the structure of the ontology. Let's assume that 'Ibuprofen' is a concept of the ontology and that 'Anti-inflammatory' is another concept. The ontology is structured in such a way that we can have a relationship between 'Ibuprofen' and 'Anti-inflammatory' so that:

'Ibuprofen' -> 'Is a' -> 'Anti-inflammatory'

We can then imagine many other relationships between concepts. The set of all the relationships is stored in the [concept_relationship.csv](data/concept_relationship.csv) table while the set of all concepts relevant information(name, class, etc.) is stored in the [concept.csv](data/concept.csv) table.

Given that many healthcare professionals are often wary of using closed-source LLM models due to the lack of direct access to the model's training data I thought that attempting to finetune an open-source model to expand its knowledge using the hemonc.org ontology would be a good project to partake and learn from. 

Additionally, I thought it could be a good idea to use this project as an opportunity to showcase all the basic necessary steps that go into creating a simple chatbot application that doesn't use RAG or vector databases starting from raw data.

## About The Project
The repo contains:

1) Hemonc.org ontology knowledge graphs data.
   - [concept.csv](data/concept.csv)
   - [concept_relationship.csv](data/concept_relationship.csv)

2) Code to clean the data and prepare it for training.
   - [data_prep](data_prep/)
     - code is available both in python script formats(recommended) and Jupyter notebooks. Either can be used.
     - Because this data prep pipeline was coded without the intention of being shared to the broader public, there are multiple scripts that should be run manually. Ideally in the future I will create a utility to run the scripts in the correct order and better document the process. Having said so, here is a brief description of the process:
        - First run [encode_knowledge_graph.py](data_prep/script/encode_knowledge_graph.py) or [encode_knowledge_graph.ipynb](data_prep/notebook/encode_knowledge_graph.ipynb) to encode the knowledge graph into text.
        - Then run [generate_qa.py](data_prep/script/generate_qa.py) or [generate_qa.ipynb](data_prep/notebook/generate_qa.ipynb) to generate the Q&A pairs using an LLM of choice. I used claude haiku due to the simplicity of the task. For something like this, you can also use open source models if you want to run them on your own hardware.
        - Then run [extract_qa.py](data_prep/script/extract_qa.py) or [extract_qa.ipynb](data_prep/notebook/extract_qa.ipynb) to extract the Q&A pairs from the LLM output of the previous step.
        - Then run [clean_qa.py](data_prep/script/clean_qa.py) or [clean_qa.ipynb](data_prep/notebook/clean_qa.ipynb) to clean the Q&A pairs. Sometimes the LLM will generate Q&A pairs that are not in the right format or just not looking good for training. This script will remove them.
        - Then run [generate_eval_qa.py](data_prep/script/generate_eval_qa.py) or [generate_eval_qa.ipynb](data_prep/notebook/generate_eval_qa.ipynb) to generate evaluation Q&A pairs based on the training Q&A pairs.
        - Then run [extract_eval_qa.py](data_prep/script/extract_eval_qa.py) or [extract_eval_qa.ipynb](data_prep/notebook/extract_eval_qa.ipynb) to extract the evaluation Q&A pairs from the LLM output of the previous step.
        - Then run [clean_qa.py](data_prep/script/clean_qa.py) or [clean_qa.ipynb](data_prep/notebook/clean_qa.ipynb) to clean the evaluation Q&A pairs. If you do, make sure you save to an evaluation dataset.
        - This process is based on [GLaM: Fine-Tuning Large Language Models for Domain Knowledge Graph](https://arxiv.org/pdf/2402.06764).
    - Generated Datasets:
      - Training dataset: 
        - [training_data](data/training_data.pkl) - first version of the training data generated from the ontology.
        - [augmented_training_data](data/augmented/augmented_training_data.pkl) - second version of training data generated from the ontology by re-running the data prep scripts multiple times with different model prompts and combining the outputs together. I recommend using this version of the training data for finetuning.
      - Testing dataset:
        - [eval_data](data/eval_data.pkl) - testing data used during the training process to evaluate the model's performance. It's directly obtained from the training data which isn't optimal but good enough for the educational purposes of this repo.
3) Code to train a Llama-2-7b-chat-hf model.
   - [train.py](train/train.py) - script used for training the model on [RunPod](https://docs.runpod.io/).
   - [train_QLORA.ipynb](train/train_QLORA.ipynb) - sample code to train the model with QLoRA on google colab.
4) A simple app to chat with the model. The app is deployed at <insert link here>
   - [Model](https://huggingface.co/GiorgioDiSalvo/Llama-2-7b-hemonchat-v1) - deployed using HuggingFace Inference Endpoints.
   - [backend](app/backend/) - the backend of the app.
   - [frontend](app/frontend/) - the frontend of the app.

<!-- CONTACT -->
## Contact

Giorgio Di Salvo - disalvogiorgio97@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/PerifanosPrometheus/HemOncOntologyMadeEasy.svg?style=for-the-badge
[contributors-url]: https://github.com/PerifanosPrometheus