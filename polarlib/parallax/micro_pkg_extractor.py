from textblob import TextBlob

import pickle, networkx as nx, numpy

from nltk.corpus import stopwords
from scipy.spatial.distance import cdist
from polarlib.prism.polarization_knowledge_graph import *

from polarlib.utils import *

from polarlib.polar.attitude.syntactical_sentiment_attitude import SyntacticalSentimentAttitudePipeline
from polarlib.polar.news_corpus_collector import *
from polarlib.polar.actor_extractor import *
from polarlib.polar.coalitions_and_conflicts import *
from polarlib.polar.sag_generator import *

from sentence_transformers import SentenceTransformer

class MicroPKGExtractor:

    def __init__(self, pkg:PolarizationKnowledgeGraph, nlp, mpqa_path):

        self.pkg        = pkg
        self.nlp        = nlp
        self.topic_dict = self.pkg.topic_dict
        self.mpqa_path  = mpqa_path

        self.model      = SentenceTransformer('all-mpnet-base-v2')

        self.hyphen_regex        = r'(?=\S+[-])([a-zA-Z-]+)'
        self.english_stopwords = stopwords.words('english')
        self.topic_list          = list(self.topic_dict)
        self.topic_centroid_dict = {}

        all_noun_phrases = set()

        for t in self.topic_dict:
            all_noun_phrases.update(self.topic_dict[t]['pre_processed'])

        BATCH_SIZE   = 1024
        encoded_dict = {}

        total_batches = list(to_chunks(list(all_noun_phrases), BATCH_SIZE))

        for batch in tqdm(total_batches, desc="Encoding batches"):

            encoded_batch = self.model.encode(batch)
            for np, vec in zip(batch, encoded_batch): encoded_dict[np] = vec

        self.topic_centroid_dict = {}

        for t in tqdm(self.topic_list):

            vectors = [encoded_dict[np] for np in set(self.topic_dict[t]['pre_processed'])]

            if   len(vectors) == 0: continue
            elif len(vectors) == 1: centroid = vectors[0]
            else: centroid = numpy.mean(vectors, axis=0)

            self.topic_centroid_dict[t] = centroid

        # for t in tqdm(self.topic_list):
        #
        #     v = self.model.encode(set(self.topic_dict[t]['pre_processed']))
        #     c = numpy.mean(v, axis=0)
        #
        #     self.topic_centroid_dict[t] = c

        self.pkg_entity_set = []

        for n, v in dict(self.pkg.pkg.nodes(data=True)).items():
            if v['type'] == 'Entity':
                self.pkg_entity_set.append(n)

    def _tokenize(self, text):
        return nltk.word_tokenize(text)

    def _remove_trailing(self, text):
        return text.strip()

    def _reduce_white_space(self, text):
        return re.sub(' +', ' ', text)

    def _to_lower_case(self, text):
        return text.lower()

    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.english_stopwords]

    def _remove_digit_tokens(self, tokens):
        return [t for t in tokens if not all(c.isdigit() for c in t)]

    def _remove_punctuation(self, text):
        _text = []

        for t in self._tokenize(text):
            if not len(re.findall(self.hyphen_regex, t)) > 0:
                t = ''.join(c if c not in string.punctuation else ' ' for c in t)
            else:
                hyphen_parts = t.split('-')
                hyphen_parts = [self._remove_punctuation(_) for _ in hyphen_parts]
                t = '-'.join(hyphen_parts)

            t = t.strip()
            if len(t) > 0: _text.append(t)

    def _remove_punctuation(self, text):
        _text = []

        for t in self._tokenize(text):
            if not len(re.findall(self.hyphen_regex, t)) > 0:
                t = ''.join(c if c not in string.punctuation else ' ' for c in t)
            else:
                hyphen_parts = t.split('-')
                hyphen_parts = [self._remove_punctuation(_) for _ in hyphen_parts]
                t = '-'.join(hyphen_parts)

            t = t.strip()
            if len(t) > 0: _text.append(t)

        return ' '.join(_text)

    def _lemmatize(self, np):
        blob = TextBlob(np)
        tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
        word_tag_list = [(w, tag_dict.get(pos[0], 'n')) for w, pos in blob.tags]

        return " ".join([w.lemmatize(t) for w, t in word_tag_list])

    def _pipeline_func(self, text, func_list):
        for f in func_list: text = f(text)
        return text

    def _extract_article_info(self, text, uid, output_dir = None, bin_category_mapping=None, verbose=False):

        if bin_category_mapping is None:

            bin_category_mapping = {
                "NEGATIVE": [(-1.000000, 0.000000)],
                "NEUTRAL":  [(0.000000, 0.000001)],
                "POSITIVE": [(0.000001, 1.000000)]
            }

        out = os.path.join(output_dir, f'articles/{uid}/')

        os.makedirs(out, exist_ok=True)

        with open(os.path.join(out, f'{uid}.json'), 'w') as f: json.dump({
            'uid':   uid,
            'text':  text
        }, f)

        if verbose: print('Pre-processing article.')

        t0 = time.time()

        corpus_collector = NewsCorpusCollector(
            output_dir   = output_dir,
            from_date    = date(year = 2023, month = 8, day = 1),
            to_date      = date(year = 2023, month = 8, day = 25),
            keywords     = []
        )

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        corpus_collector.pre_process_article(os.path.join(out, f'{uid}.json'))

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if verbose: print('Identifying entities.')

        t0 = time.time()

        entity_extractor = EntityExtractor(
            output_dir = output_dir,
            entity_set = set(self.pkg_entity_set)
        )

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if len(entity_extractor.article_paths) == 0:
            print('`Entity Extractor` couldn\'t locate a file to parse.')
            return None

        entity_extractor.extract_article_entities(entity_extractor.article_paths[0])

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if verbose: print('Identifying noun phrases.')

        t0 = time.time()

        noun_phrase_extractor = NounPhraseExtractor(output_dir = output_dir)

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if len(noun_phrase_extractor.entity_paths) == 0:
            print('`Noun Phrase Extractor` couldn\'t locate a file to parse.')
            return None

        noun_phrase_extractor.extract_article_noun_phrases(noun_phrase_extractor.entity_paths[0])

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if verbose: print('Extracting sentiment attitudes.')

        t0 = time.time()

        sentiment_attitude_pipeline = SyntacticalSentimentAttitudePipeline(
            output_dir  = output_dir,
            nlp         = self.nlp,
            mpqa_path   = self.mpqa_path
        )

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if len(sentiment_attitude_pipeline.noun_phrase_path_list) == 0:
            print('`Sentiment Attitude Pipeline` couldn\'t locate a file to parse.')
            return None

        sentiment_attitude_pipeline.extract_sentiment_attitude(
            sentiment_attitude_pipeline.noun_phrase_path_list[0]
        )

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if verbose: print('Generating SAG.')

        t0 = time.time()

        sag_generator = SAGGenerator(output_dir)

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        if len(sag_generator.attitude_path_list) == 0:
            print('`SAG Generator` couldn\'t locate a file to parse.')
            return None

        sag_generator._read_sentiment_attitudes(sag_generator.attitude_path_list[0])

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

        sag_generator.convert_attitude_signs(bin_category_mapping = bin_category_mapping)

        sag_generator.construct_sag()

        t1 = time.time()

        if verbose: print('Done.', t1 - t0)

    def _identify_article_topics(self, noun_phrase_list):

        clean_noun_phrase_dict = {
            np: self._pipeline_func(np, [
                self._lemmatize,
                self._to_lower_case,
                self._remove_punctuation,
                self._remove_trailing,
                self._reduce_white_space,
                self._tokenize,
                self._remove_digit_tokens,
                self._remove_stopwords,
                lambda t: ' '.join(t)
            ]) for np in tqdm(noun_phrase_list)
        }

        clean_np_list = [clean_noun_phrase_dict[np] for np in noun_phrase_list]

        if len(clean_np_list) == 0: return {}

        vectors       = self.model.encode(clean_np_list)

        topic_identifier_list = list(self.topic_centroid_dict)

        distance_matrix = cdist(vectors, [self.topic_centroid_dict[t] for t in topic_identifier_list], metric='cosine')

        np_to_topic = {np: [] for np in noun_phrase_list}

        for i, d in enumerate(distance_matrix):

            t_indices = numpy.where(d <= 0.2)[0]

            np_to_topic[noun_phrase_list[i]] += [topic_identifier_list[_] for _ in t_indices]

        return np_to_topic

    @staticmethod
    def convert_sentiment_attitude(sentiment_value, sentiment_mapping):

        sentiment_category = "NEUTRAL"

        for category, category_bins in sentiment_mapping.items():

            for b in category_bins:

                if sentiment_value >= b[0] and sentiment_value < b[1]:
                    sentiment_category = category
                    break

        return sentiment_category

    @staticmethod
    def _get_neighbors(G, node, attr_label=None, attr_value=None):

        neighbors = list(G.neighbors(node))

        if attr_label == None: return neighbors

        else: return [neighbor for neighbor in neighbors if G.nodes(data=True)[neighbor].get(attr_label) == attr_value]

    def encode_micro_pkg(self, text, uid, output_dir = None, bin_category_mapping=None, verbose=False):

        if bin_category_mapping is None:

            bin_category_mapping = {
                "NEGATIVE": [(-1.000000, 0.000000)],
                "NEUTRAL":  [(0.000000, 0.000001)],
                "POSITIVE": [(0.000001, 1.000000)]
            }

        if verbose: print('Extracting article information.')

        self._extract_article_info(text, uid, output_dir, bin_category_mapping)

        if verbose: print('Done.')

        if not os.path.exists(f"{output_dir}/attitudes/{uid}/{uid}.pckl"): return None

        with open(f"{output_dir}/attitudes/{uid}/{uid}.pckl", 'rb') as f: article_attitudes = pickle.load(f)

        article_noun_phrase_list = []

        for s in article_attitudes['attitudes']:

            for kv in s['noun_phrase_attitudes']:

                article_noun_phrase_list.append(kv[1])

        article_noun_phrase_list = list(set(article_noun_phrase_list))

        if verbose: print('Identifying article topics.')

        np_to_topic = self._identify_article_topics(article_noun_phrase_list)

        if verbose: print('Done.')

        if verbose: print('Constructing micro-PKG.')

        micro_pkg = nx.DiGraph()

        entity_topic_attitude_dict = {}
        article_topic_list         = []

        for s in article_attitudes['attitudes']:

            for kv in s['entity_attitudes']:

                micro_pkg.add_node(kv[0], type='Entity')
                micro_pkg.add_node(kv[1], type='Entity')

                att = numpy.mean(s['entity_attitudes'][kv])
                if att == 0: continue

                micro_pkg.add_edge(kv[0], kv[1], weight=att, type='Relationship', label='NEGATIVE' if att < 0 else 'POSITIVE')
                micro_pkg.add_edge(kv[1], kv[0], weight=att, type='Relationship', label='NEGATIVE' if att < 0 else 'POSITIVE')

            for kv in s['noun_phrase_attitudes']:

                np = kv[1]

                if np not in np_to_topic: continue

                if kv[0] not in entity_topic_attitude_dict: entity_topic_attitude_dict[kv[0]] = {}

                for t in np_to_topic[kv[1]]:

                    if t not in entity_topic_attitude_dict[kv[0]]: entity_topic_attitude_dict[kv[0]][t] = []

                    entity_topic_attitude_dict[kv[0]][t] += s['noun_phrase_attitudes'][kv]

            for e in entity_topic_attitude_dict:

                micro_pkg.add_node(e, type='Entity')

                for t in entity_topic_attitude_dict[e]:

                    micro_pkg.add_node(t, type='Topic')

                    att = numpy.mean(entity_topic_attitude_dict[e][t])
                    lbl = self.convert_sentiment_attitude(att, bin_category_mapping)

                    article_topic_list.append(t)
                    micro_pkg.add_edge(e, t, type='Attitude', weight=att, label=lbl)

            article_entity_set     = []
            article_fellowship_set = []
            article_dipole_set     = []

            for n, v in dict(micro_pkg.nodes(data=True)).items():

                if v['type'] == 'Entity': article_entity_set.append(n)

            for e in article_entity_set:

                article_fellowship_set += self._get_neighbors(
                    self.pkg.pkg,
                    e,
                    attr_label='type',
                    attr_value='Fellowship'
                )

            for f12 in itertools.combinations(article_fellowship_set, 2):

                d_label_1 = 'D' + f12[0].replace('F', '') + '_' + f12[1].replace('F', '')
                d_label_2 = 'D' + f12[1].replace('F', '') + '_' + f12[0].replace('F', '')

                d1 = self._get_neighbors(
                    self.pkg.pkg,
                    f12[0],
                    attr_label='type',
                    attr_value='Dipole'
                )

                if d_label_1 in d1: article_dipole_set.append(d_label_1)
                if d_label_2 in d1: article_dipole_set.append(d_label_2)

            for e in self.pkg.pkg.subgraph(
                    article_topic_list + article_entity_set + article_fellowship_set + article_dipole_set
            ).edges(data=True):

                n1    = e[0]
                n2    = e[1]
                attrs = e[2]

                if self.pkg.pkg.nodes[n1] == 'Topic'  or self.pkg.pkg.nodes[n2]  == 'Topic':  continue
                if self.pkg.pkg.nodes[n1] == 'Entity' and self.pkg.pkg.nodes[n2] == 'Entity': continue

                micro_pkg.add_edge(*e[:2], **e[2])

            if verbose: print('Done.')

            return micro_pkg



