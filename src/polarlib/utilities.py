import json

def load_article(path, func=lambda t: json.loads(json.load(t))):
    with open(path, 'r') as f: article_obj = func(f)
    return article_obj

def jaccard_index(s1, s2): return len(set(s1).intersection(set(s2))) / len(set(s1).union(set(s2)))

def to_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]