import os, spacy, itertools, requests, time, string, spotlight, nltk, multiprocessing

from .miscellaneous import *

from tqdm import tqdm
from nltk import ngrams
from multiprocessing import Pool

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from spotlight import SpotlightException

from mosestokenizer import MosesTokenizer, MosesDetokenizer

class EntityExtractor:
    """
    A class for extracting entities from articles and querying DBpedia for entity information.

    Args:
        output_dir (str): The directory where the output data will be stored.

    Attributes:
        output_dir (str): The output directory for storing extracted entity data.
        article_paths (list): List of article file paths obtained from the 'pre_processed' folder.
    """

    def __init__(self, output_dir):
        """
        Initialize the EntityExtractor.

        Args:
            output_dir (str): The directory where the output data will be stored.
        """
        self.output_dir      = output_dir

        self.article_paths   = list(itertools.chain.from_iterable([
            [os.path.join(o1, p) for p in o3]
            for o1, o2, o3 in os.walk(os.path.join(self.output_dir, 'pre_processed'))
        ]))

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

        time.sleep(0.250)

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

            output_folder = os.path.join(self.output_dir, 'entities/' + path.split('/')[-2])
            output_file   = os.path.join(output_folder, article['uid'] + '.json')

            if os.path.exists(output_file): return True

            sentence_list = article['text'].split('\n')
            sentence_list = [sent_tokenize(s) for s in sentence_list]
            sentence_list = list(itertools.chain.from_iterable(sentence_list))
            entity_list   = self.query_dbpedia_entities(article['text'])

            max_from_i, sentence_object_list = 0, []

            for s in sentence_list:

                from_i, to_i = article['text'][max_from_i:].index(s), len(s)

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

                for e in entity_list:

                    e_range_set = set(list(range(e['begin'], e['end'])))

                    if len(s_range_set.intersection(e_range_set)) > 0: sentence_object['entities'].append(e)

                sentence_object_list.append(sentence_object.copy())

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

    def extract_entities(self):
        """
        Extract entities from all articles using multiprocessing.

        This method uses multiprocessing to extract entities from multiple articles concurrently.
        """
        pool = Pool(multiprocessing.cpu_count() - 8)

        for i in tqdm(
                pool.imap_unordered(self.extract_article_entities, self.article_paths),
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

            print(ex)
            return None

        return True

    def extract_noun_phrases(self):
        """
        Extract noun phrases from all articles using multiprocessing.
        """
        pool = Pool(multiprocessing.cpu_count() - 8)

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

    nlp = spacy.load("en_core_web_sm")

    entity_extractor = EntityExtractor(output_dir   = "../example")

    entity_extractor.extract_entities()

    noun_phrase_extractor = NounPhraseExtractor(output_dir="../example")

    noun_phrase_extractor.extract_noun_phrases()

