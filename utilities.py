from collections import defaultdict
from enum import Enum, auto as enum_auto
import json

def get_text(tokens, after=None):
    raw_text = []
    for token in tokens:
        raw_text.append(token['originalText'])
        raw_text.append(after or token.get('after', ' '))
        
    return ''.join(raw_text[:-1])

sentences_from_parsed = lambda sentences: [
    {
        'index': s_idx,
        'tokens': [
            {
                'index': t_idx + 1,
                'originalText': token
            } for t_idx, token in enumerate(sentence)
        ]
    } for s_idx, sentence in enumerate(sentences)
]

chains_from_parsed = lambda chains: [
    [
        {
            'sentNum': m['sent_ind'],
            'startIndex': m['token_ind'],
            'endIndex': m['end_ind']
        } for m in chain
    ] for chain in chains
]

def load_from_parsed(f):
    parsed_data = json.load(f)
    sentences = sentences_from_parsed(parsed_data['text'])
    corefs = chains_from_parsed(parsed_data['cluster_json'])
    return parsed_data, sentences, corefs

def load_from_text(f):
    raw_text = f.read()
    nlp = StanfordCoreNLP('http://localhost:9000')
    output = nlp.annotate(raw_text, properties={
        'annotators': 'ner,coref',
        'outputFormat': 'json'
    })
    sentences = output['sentences']
    corefs = output['corefs'].values()
    return output, sentences, corefs

def write_deps_to_tokens(sentences):
    for sentence in sentences:
        for dep in sentence['enhancedPlusPlusDependencies']:
            gov_idx = dep['governor'] - 1
            dep_idx = dep['dependent'] - 1
            dep_type = dep['dep'].lower()
            if dep_type == 'root': sentence['dep_root'] = dep['dependent']
            else:
                gov = sentence['tokens'][gov_idx]
                if 'dependents' not in gov: gov['dependents'] = set()
                gov['dependents'].add((dep_idx, dep_type))
            if 'governor' not in sentence['tokens'][dep_idx]: sentence['tokens'][dep_idx]['governor'] = set()
            sentence['tokens'][dep_idx]['governor'].add((gov_idx, dep_type))

class DepDirection(Enum):
    GOV = enum_auto()
    DEP = enum_auto()

def find_dep_path(tokens, source_idx, dest_idx):
    visited = set()
    q = []
    q.append((source_idx, [((None, None), source_idx)]))
    curr, path = q.pop(0)
    while curr != dest_idx:
        visited.add(curr)
        
        for gov, gov_type in tokens[curr]['governor']:
            if gov_type != 'root' and gov not in visited: q.append((gov, path + [((DepDirection.GOV, gov_type), gov)]))
        for dep, dep_type in tokens[curr]['dependents'] if 'dependents' in tokens[curr] else []:
            if dep not in visited: q.append((dep, path + [((DepDirection.DEP, dep_type), dep)]))
        if len(q) == 0: return None
        curr, path = q.pop(0)
    return path

def dep_path_to_token_slice(sentence, dep_path):
    t_slice = []
    for _, idx in dep_path: t_slice.append(sentence[idx])
    return t_slice