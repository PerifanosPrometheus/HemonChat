import re

def extract_qa(text):
    q_pattern = re.compile(r'Q[0-9]:(.*?\?)', re.MULTILINE)
    a_pattern = re.compile(r'^\s*A\d*:\s*(.+?)(?=^\s*Q\d+:|\Z)', re.MULTILINE | re.DOTALL)
    questions = re.findall(q_pattern, text)
    answers = re.findall(a_pattern,text)
    if len(questions)!=len(answers):
        qa_list = -1
    else:
        qa_list = [{'question':q.strip(),'answer':a.strip()} for q,a in zip(questions,answers)]
    return qa_list

def extract_eval_qa(text):
    q_pattern = re.compile(r'^\s*Q[0-9]:(.*?)(?=^\s*A\d*:|\Z)', re.MULTILINE | re.DOTALL)
    a_pattern = re.compile(r'^\s*A\d*:\s*(.+?)(?=^\s*Q\d+:|\Z)', re.MULTILINE | re.DOTALL)
    questions = re.findall(q_pattern, text)
    answers = re.findall(a_pattern,text)
    if len(questions)!=len(answers):
        raise Exception(f"LLM response does not follow expected format:\n{text}")
    else:
        qa_list = [{'question':q.strip(),'answer':a.strip()} for q,a in zip(questions,answers)]
    return qa_list

def is_acceptable_answer(text):
    is_acceptable=True
    not_acceptable_pattern = re.compile(r'(context|text|information).*does not')
    if not_acceptable_pattern.search(text):
        is_acceptable=False
    return is_acceptable