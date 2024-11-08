from multiprocessing import Pool

import itertools, json
import nltk
import os
import requests
import spacy
import string
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from spotlight import SpotlightException
from tqdm import tqdm

from collections import defaultdict
from polarlib.utils.utils import *

from fastcoref import LingMessCoref as OriginalLingMessCoref
from fastcoref import FCoref as OriginalFCoref
from transformers import AutoModel
import functools

class PatchedLingMessCoref(OriginalLingMessCoref):
    def __init__(self, *args, **kwargs):
        original_from_config = AutoModel.from_config

        def patched_from_config(config, *args, **kwargs):
            kwargs['attn_implementation'] = 'eager'
            return original_from_config(config, *args, **kwargs)

        try:
            AutoModel.from_config = functools.partial(patched_from_config, attn_implementation='eager')
            super().__init__(*args, **kwargs)
        finally:
            AutoModel.from_config = original_from_config

class PatchedFCoref(OriginalFCoref):
    def __init__(self, *args, **kwargs):
        original_from_config = AutoModel.from_config

        def patched_from_config(config, *args, **kwargs):
            kwargs['attn_implementation'] = 'eager'
            return original_from_config(config, *args, **kwargs)

        try:
            AutoModel.from_config = functools.partial(patched_from_config, attn_implementation='eager')
            super().__init__(*args, **kwargs)
        finally:
            AutoModel.from_config = original_from_config      

def align_clusters_to_char_level(clusters, char_map):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for start, end in cluster:
            span_idx, span_char_level = char_map[(start, end)]
            if span_char_level is None:
                continue
            new_cluster.append(span_char_level)
        new_clusters.append(new_cluster)
    return new_clusters

def resolve_clusters(text, clusters):
    """
    Replaces all mentions in each cluster with the content of the first mention in that cluster.
    
    Parameters:
        text (str): The original text.
        clusters (list of list of tuples): A list where each cluster is a list of mention spans (start, end).
        
    Returns:
        str: The text with mentions in each cluster replaced by the content of the first mention.
    """
    
    replacements = []
    for cluster in clusters:
        
        main_text = text[cluster[0][0]:cluster[0][1]]
        
        for mention in cluster[1:]:
            start, end = mention
            replacements.append((start, end, main_text))
    
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    resolved_text = text
    for start, end, replacement in replacements:
        resolved_text = resolved_text[:start] + replacement + resolved_text[end:]
    
    return resolved_text

def remove_duplicate_entities(data):
    unique_entities = []
    seen_entities = set()
    
    for entity in data['entities']:
        identifier = (entity['begin'], entity['end'])
        
        if identifier not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(identifier)
    
    data['entities'] = unique_entities
    
    """
    unique_entities = []
    seen_entities = set()

    for entity in data['entities']:
        identifier = entity['title']
        
        if identifier not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(identifier)
    
    data['entities'] = unique_entities
    """

    return data

class EntityExtractor:
    """
    A class for extracting entities from articles and querying DBpedia for entity information.

    Args:
        output_dir (str): The directory where the output data will be stored.

    Attributes:
        output_dir (str): The output directory for storing extracted entity data.
        article_paths (list): List of article file paths obtained from the 'pre_processed' folder.
    """

    def __init__(self, output_dir, coref=False, entity_set=None):
        """
        Initialize the EntityExtractor.

        Args:
            output_dir (str): The directory where the output data will be stored.
        """
        self.output_dir = output_dir
        self.entity_set = entity_set
        self.spacy_nlp  = spacy.load("en_core_web_sm")

        self.article_paths   = list(itertools.chain.from_iterable([
            [os.path.join(o1, p) for p in o3]
            for o1, o2, o3 in os.walk(os.path.join(self.output_dir, 'pre_processed'))
        ]))

        self.coref_model = PatchedLingMessCoref(
            nlp    = "en_core_web_sm",
            device = "cpu"
        )

        self.coref_flag = coref

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def coreference_resolution(self, text, verbose=False):

        pred = self.coref_model.predict(texts=[text])[0]
   
        resolved_text = resolve_clusters(text, align_clusters_to_char_level(pred.clusters, pred.char_map))

        if verbose:
            print("Original Text: ", text)
            print("Resolved Text: ", resolved_text)
            print()

        return resolved_text

    def _get_named_entities(self, text):

        """ner_types = ['PERSON', 'LOC', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW']"""

        ner_types   = ['PERSON', 'NORP', 'ORG', 'GPE']

        ner_pairs = []

        doc = self.spacy_nlp(text)

        for np in doc.noun_chunks:

            for e in np.ents:

                if e.label_ in ner_types: ner_pairs.append({
                    'text': e.text,
                    'start': e.start_char,
                    'end': e.end_char
                })

        return ner_pairs

    def _get_entity_mention(self, text):

        dbpedia_mentions = defaultdict(lambda: [])

        e_list = self._get_named_entities(text)
        t      = ' and '.join([e['text'] for e in e_list])

        if len(t) == 0: return None

        dbpedia_entities = self.query_dbpedia_entities(t, confidence=0.45)

        _entities = []

        for e1 in e_list:

            e1_interval = set(range(e1['start'], e1['end']))

            for e2 in dbpedia_entities:

                e2_interval = set(range(e2['begin'], e2['end']))

                if len(e1_interval.intersection(e2_interval)) > 0: _entities.append(e2)

        return _entities

    def _get_entity_mentionv2(self, text, dbpedia_entities, sentence_start_offset=0):

        e_list = self._get_named_entities(text)

        _entities = []

        for e1 in e_list:

            e1_start_global = e1['start'] + sentence_start_offset
            e1_end_global = e1['end'] + sentence_start_offset
            e1_interval = set(range(e1_start_global, e1_end_global))

            for e2 in dbpedia_entities:

                e2_interval = set(range(e2['begin'], e2['end']))

                if len(e1_interval.intersection(e2_interval)) > 0: _entities.append(e2)

        return _entities

    def query_dbpedia_entities(self, text, confidence=0.45, spotlight_url='http://127.0.0.1:2222/rest/annotate'):
        """
        Query DBpedia Spotlight for entities in the given text.

        Args:
            text (str): The input text to query for entities.
            confidence (float, optional): Confidence threshold for entity extraction. Defaults to 0.45.
            spotlight_url (str, optional): URL of the DBpedia Spotlight service. Defaults to 'http://127.0.0.1:2222/rest/annotate'.

        Returns:
            list: A list of dictionaries representing extracted entities.
        """
        req_data      = {'lang': 'en', 'text': str(text), 'confidence': confidence, 'types': ['']}

        spot_entities = requests.post(spotlight_url, data=req_data, headers={"Accept": "application/json"})

        try:
            if 'Resources' not in spot_entities.json(): raise SpotlightException()
            spot_entities = [{k[1:]: v for k, v in r.items()} for r in spot_entities.json()['Resources']]
        except SpotlightException as se:
            print(se)
            return []

        return [{
            'begin': int(e['offset']),
            'end': int(e['offset']) + len(e['surfaceForm']),
            'title': e['URI'],
            'score': float(e['similarityScore']),
            'rank': float(e['percentageOfSecondRank']),
            'text': e['surfaceForm'],
            'types': [t.replace('Wikidata:', '') for t in e['types'].split(',') if 'Wikidata' in t],
            'wikid': e['URI'],
            'dbpedia': e['URI']
        } for e in spot_entities]

    def extract_entities_from_text(self, text, confidence=0.40, coref=False):
        """
        Extract entities from a single article.

        Args:
            path (str): The path to the article file.

        Returns:
            bool or None: Returns True if extraction is successful, None if an exception occurs.
        """

        try:

            sentence_list = text.split('\n')
            sentence_list = [sent_tokenize(s) for s in sentence_list]
            sentence_list = list(itertools.chain.from_iterable(sentence_list))

            entity_list   = self.query_dbpedia_entities(text, confidence=confidence)

            max_from_i, sentence_object_list = 0, []

            for s in sentence_list:

                from_i, to_i = text[max_from_i:].index(s), len(s)

                from_i     += max_from_i
                to_i       += from_i
                max_from_i  = to_i

                sentence_object = {
                    "sentence": s,
                    "from":     from_i,
                    "to":       to_i,
                    "entities": []
                }

                s_range_set = set(list(range(from_i, to_i)))

                s_entity_list = self._get_entity_mentionv2(s, entity_list, from_i)

                for e in s_entity_list:

                    if e == None or (self.entity_set and e['title'] not in self.entity_set): continue

                    e_range_set = set(list(range(e['begin'], e['end'])))

                    if len(s_range_set.intersection(e_range_set)) > 0: sentence_object['entities'].append(e)

                sentence_object_list.append(sentence_object.copy())

        except Exception as ex:
            print(ex)
            return None
        
        if coref:
                
            pred = self.coref_model.predict(texts=[text])[0]

            coreference_char_clusters = align_clusters_to_char_level(pred.clusters, pred.char_map)

            char_sentence_indices = {}

            for i, s in enumerate(sentence_object_list):

                for c in range(s['from'], s['to']):

                    char_sentence_indices[c] = i
                    
            new_entities = [[] for s in sentence_object_list]

            for i, sentence in enumerate(sentence_object_list):

                for e in sentence['entities']:

                    entity_range = set(list((e['begin'], e['end'])))

                    for c in coreference_char_clusters:
                    
                        origin_c     = c[0]
                        origin_range = set(list(range(origin_c[0], origin_c[1])))
                    
                        if len(origin_range.intersection(entity_range)) > 0: 

                            for k in c[1:]:
                                
                                new_entities[char_sentence_indices[k[0]]].append({
                                    "begin": k[0],
                                    "end":   k[1],
                                    "title": e['title'],
                                    "score": e['score'],
                                    "rank":  e['rank'],
                                    "text":  text[k[0]:k[1]],
                                    "types": e['types'],
                                    "wikid": e['wikid'],
                                    "dbpedia": e['dbpedia']
                                })

            for i, es in enumerate(new_entities):

                sentence_object_list[i]['entities'] += es

            """The code below is to remove duplicate mentions, both at the indices level and at the title level."""
            sentence_object_list = [remove_duplicate_entities(d) for d in sentence_object_list]

        return sentence_object_list

    def extract_article_entities(self, path):
        """
        Extract entities from a single article.

        Args:
            path (str): The path to the article file.

        Returns:
            bool or None: Returns True if extraction is successful, None if an exception occurs.
        """
        try:

            article = load_article(path)
            text    = article['text']

            output_folder = os.path.join(self.output_dir, 'entities/' + path.split('/')[-2])
            output_file   = os.path.join(output_folder, article['uid'] + '.json')

            if os.path.exists(output_file): return True

            """
            sentence_list = text.split('\n')
            sentence_list = [sent_tokenize(s) for s in sentence_list]
            sentence_list = list(itertools.chain.from_iterable(sentence_list))

            entity_list   = self.query_dbpedia_entities(text)

            max_from_i, sentence_object_list = 0, []

            for s in sentence_list:

                from_i, to_i = text[max_from_i:].index(s), len(s)

                from_i     += max_from_i
                to_i       += from_i
                max_from_i  = to_i

                sentence_object = {
                    "sentence": s,
                    "from":     from_i,
                    "to":       to_i,
                    "entities": []
                }

                s_range_set = set(list(range(from_i, to_i)))

                s_entity_list = self._get_entity_mentionv2(s, entity_list)

                for e in s_entity_list:

                    if e == None or (self.entity_set and e['title'] not in self.entity_set): continue

                    e_range_set = set(list(range(e['begin'], e['end'])))

                    if len(s_range_set.intersection(e_range_set)) > 0: sentence_object['entities'].append(e)

                sentence_object_list.append(sentence_object.copy())
            """

            sentence_object_list = self.extract_entities_from_text(text, coref=self.coref_flag)

            article_dict_str = json.dumps({
                'uid': article['uid'],
                'entities': sentence_object_list.copy()
            })

            if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
            with open(output_file, 'w') as f:     json.dump(article_dict_str, f)

        except Exception as ex:
            print(ex)
            return None

        return True

    def extract_entities(self, n_processes=32):
        """
        Extract entities from all articles using multiprocessing.

        This method uses multiprocessing to extract entities from multiple articles concurrently.
        """

        if n_processes == 1:

            for p in tqdm(self.article_paths, desc  = 'Identifying Article Entities'):

                self.extract_article_entities(p)

        else:

            pool = Pool(n_processes)

            for i in tqdm(
                    pool.map(self.extract_article_entities, self.article_paths),
                    desc  = 'Identifying Article Entities',
                    total = len(self.article_paths)
            ): pass

            pool.close()
            pool.join()

class NounPhraseExtractor:
    """
    A class for extracting noun phrases from articles.
    """

    nltk.download('brown')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

    def __init__(self, output_dir, spacy_model_str="en_core_web_sm"):
        """
        Initialize the NounPhraseExtractor.

        :param output_dir: The directory to output the results.
        :param spacy_model_str: The Spacy model to use for sentence parsing.
        """
        self.entity_paths  = []
        self.output_dir    = output_dir
        self.stopword_list = stopwords.words('english') + ["'s"]
        self.nlp           = spacy.load(spacy_model_str)

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'entities')):

            for p in files: self.entity_paths.append(os.path.join(root, p))

    def _clean_text(self, text):
        """
        Clean and preprocess the given text.

        :param text: The input text.
        :return: Cleaned and preprocessed text.
        """
        token_list = word_tokenize(text.lower())
        token_list = [t for t in token_list if not all(c in string.punctuation for c in t)]
        token_list = [t for t in token_list if not t in self.stopword_list]
        token_list = [t for t in token_list if len(t.strip()) > 0]

        return ' '.join(token_list)

    def extract_article_noun_phrases(self, path):
        """
        Extract noun phrases from an article.

        :param path: The path to the article.
        :return: True if successful, None otherwise.
        """
        try:

            entity_object = load_article(path)

            output_folder = os.path.join(self.output_dir, 'noun_phrases/' + path.split('/')[-2] + '/')
            output_file   = os.path.join(output_folder, entity_object['uid'] + '.json')

            if os.path.exists(output_file): return True

            noun_phrase_object = {
                'uid':          entity_object['uid'],
                'noun_phrases': entity_object['entities'].copy()
            }

            for i, s in enumerate(entity_object['entities']):

                s['sentence'] = " ".join(s['sentence'].split())

                entity_object['entities'][i]['sentence'] = s['sentence']

                n_gram_list = [
                    {
                        "ngram": np.text,
                        "from":  np.start_char + s['from'],
                        "to":    np.end_char + s['from']
                    } for np in self.nlp(s['sentence']).noun_chunks
                ]

                n_gram_list = [n for n in n_gram_list if len(self._clean_text(n['ngram'])) > 0]

                noun_phrase_object['noun_phrases'][i]['noun_phrases'] = []

                for n in n_gram_list:

                    n_range_set = set(list(range(n['from'], n['to'])))

                    entity_flag = False

                    for e in s['entities']:

                        e_range_set = set(list(range(e['begin'], e['end'])))

                        if len(e_range_set.intersection(n_range_set)) > 0:
                            entity_flag = True

                            break

                    if not entity_flag: noun_phrase_object['noun_phrases'][i]['noun_phrases'].append(n)

            article_dict_str = json.dumps(noun_phrase_object.copy())

            if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
            with open(output_file, 'w') as f:     json.dump(article_dict_str, f)

        except Exception as ex:

            print('Error found.')
            print(ex)
            return None

        return True

    def extract_noun_phrases(self, n_processes=16):
        """
        Extract noun phrases from all articles using multiprocessing.
        """
        pool = Pool(n_processes)

        for i in tqdm(
                pool.imap_unordered(self.extract_article_noun_phrases, self.entity_paths),
                desc  = 'Identifying Article Noun Phrases',
                total = len(self.entity_paths)
        ): pass

        pool.close()
        pool.join()

    def _extract_ngrams(self, s, offset=0, n=1):
        """
        Extract n-grams from a given sentence.

        :param s: The sentence to extract n-grams from.
        :param offset: Offset for position indices.
        :param n: The n-gram size.
        :return: List of extracted n-grams.
        """

        ms = MosesTokenizer('en')
        md = MosesDetokenizer('en')

        token_list  = word_tokenize(s)
        n_grams     = list(ngrams(token_list, n))
        max_from_i  = 0
        n_gram_list = []

        for grams in n_grams:

            grams = list(grams)
            if all(c in string.punctuation for c in ''.join(grams)): continue
            if all(n in self.stopword_list for n in grams):          continue

            from_i, to_i = s[max_from_i:].index(grams[0]), s[max_from_i:].index(grams[-1]) + len(grams[-1])

            from_i += max_from_i
            to_i += from_i

            n_gram_list.append({
                'ngram': md(grams),
                'from':  from_i + offset,
                'to':    to_i + offset
            })

            max_from_i = from_i

        return n_gram_list

if __name__ == "__main__":

    """python -m spacy download en_core_web_sm"""

    entity_extractor = EntityExtractor(output_dir   = "../example")

    print(
        json.dumps(
            entity_extractor.extract_article_entities(
                "/home/dpasch01/notebooks/PARALLAX/Infodemic/Coronavirus/pre_processed/20210525/www.state-journal.com.-news-coronavirus-pandemic-after-more-than-a-year-of-covid-government-meetings-will-be-in-person-soo.json"
            ),
            indent=4
        )
    )

    "entity_extractor.extract_entities()"

    """
    noun_phrase_extractor = NounPhraseExtractor(output_dir="../example")

    noun_phrase_extractor.extract_noun_phrases()
    """
