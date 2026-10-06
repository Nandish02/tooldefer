"""Shared utilities: JSONL I/O, JSON extraction, ground truth definitions."""

import json
import re
import os
from typing import List, Optional


GROUND_TRUTHS = {
    'c1_true_statement_ans_t1': '1',
    'c1_false_statement_ans_t1': '2',
    'c2_correct_chat_ans_t1': '1',
    'c2_incorrect_chat_ans_t1': '2',
}

CONDITION_NAMES = {
    'c1_true_statement_ans_t1': 'C1_True (factual+correct)',
    'c1_false_statement_ans_t1': 'C1_False (factual+wrong)',
    'c2_correct_chat_ans_t1': 'C2_Correct (dialogue+correct)',
    'c2_incorrect_chat_ans_t1': 'C2_Incorrect (dialogue+wrong)',
}

COLUMN_MAPPINGS = [
    ('c1_true_statement_ans_t1', 'c1_true_statement_t1_history', 'c1_true_statement_reasoning_t1'),
    ('c1_false_statement_ans_t1', 'c1_false_statement_t1_history', 'c1_false_statement_reasoning_t1'),
    ('c2_correct_chat_ans_t1', 'c2_correct_chat_t1_history', 'c2_correct_chat_reasoning_t1'),
    ('c2_incorrect_chat_ans_t1', 'c2_incorrect_chat_t1_history', 'c2_incorrect_chat_reasoning_t1'),
]


def load_jsonl(filepath: str) -> List[dict]:
    """Load JSONL file, return list of records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[dict], filepath: str) -> None:
    """Save records to JSONL file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def append_jsonl(record: dict, filepath: str) -> None:
    """Append single record to JSONL file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def extract_json_from_response(response: str) -> dict:
    """Extract JSON object from LLM response text."""
    if not response or not isinstance(response, str):
        return {}
    
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def extract_answer_from_text(text: str) -> tuple:
    """Extract chosen_answer from text (JSON or plain). Returns (answer, reasoning)."""
    if not text:
        return None, None
    
    # Try JSON format
    ans_match = re.search(r'"chosen_answer"\s*:\s*"?([12])"?', text)
    if ans_match:
        answer = ans_match.group(1)
        reason_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?:\s*[,}]|$)', text, re.DOTALL)
        return answer, reason_match.group(1) if reason_match else None
    
    # Try plain text (first number)
    first_num_match = re.match(r'^\s*([12])\b', text)
    if first_num_match:
        answer = first_num_match.group(1)
        rest = text[first_num_match.end():].strip()
        rest = re.sub(r'\s*[12]\s*$', '', rest).strip()
        return answer, rest if rest else None
    
    # Fallback: any standalone 1 or 2
    standalone_match = re.search(r'\b([12])\b', text)
    if standalone_match:
        return standalone_match.group(1), None
    
    return None, None


def get_assistant_response(history) -> Optional[str]:
    """Get assistant response from conversation history."""
    if not history:
        return None
    if isinstance(history, str):
        try:
            history = eval(history)
        except:
            return None
    for msg in history:
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            return msg.get('content', '')
    return None
