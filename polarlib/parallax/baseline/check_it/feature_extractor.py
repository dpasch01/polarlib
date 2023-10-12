import pandas as pd, re, numpy, nltk, json, requests, string, itertools, spacy

from os import listdir
from os.path import isfile, join
from random import shuffle

from nltk.tree import Tree
from bllipparser import RerankingParser

from string import ascii_letters, ascii_lowercase, ascii_uppercase, whitespace, punctuation, digits

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, TreebankWordTokenizer

import linguistic_inquiry_word_count
from surface import *
from readability_index import *
from vocabulary_richness import *
from psychological import *
from sentiment import *
from part_of_speech import *

from wnaffect import WNAffect
from emotion import Emotion

from nltk.corpus import wordnet
from afinn import Afinn
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')

def replace_special(text):
    text = text.replace('``', "''")
    text = text.replace('`', "'")
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace("'", "'")
    text = text.replace('–', "-")
    text = text.replace('—', "-")
    text = text.replace('\"', '"')
    text = text.replace("\'", "'")

    return text

def punctuation_sequence(text): return text.translate(str.maketrans('', '', ascii_letters + ascii_lowercase + ascii_uppercase + digits + whitespace))

def encode(text): return text.encode(encoding="ascii",errors="ignore")
def decode(text): return text.decode("utf-8")

def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text

def pipeline_func(text, func_list):
    for f in func_list: text = f(text)
    return text

class CheckItFeatureExtractor:

    def __init__(self, training_set, test_set, bllipparser_path='/home/ubuntu/.local/share/bllipparser/'):

        self.nlp          = spacy.load("en_core_web_sm")
        self.training_set = training_set
        self.test_set     = test_set

        self.training_set['training'] = [True for i in range(self.training_set.shape[0])]
        self.test_set['training'] = [False for i in range(self.test_set.shape[0])]

        self.df =  pd.concat([self.training_set, self.test_set], ignore_index=True)

        self.df['text']     = self.df['text'].apply(lambda t: pipeline_func(t, [replace_special, uncontract, lambda t: t.replace('\n', ' ')]))

        self.liwc = linguistic_inquiry_word_count.LIWC("dictionaries/LIWC2015_English_Flat.dic")

        self. liwc_categories = []

        self.rrp = RerankingParser.from_unified_model_dir(bllipparser_path)

        for c in self.liwc.lexicon.values(): self.liwc_categories = self.liwc_categories + c

        self.liwc_categories = set(self.liwc_categories)

        self.treebank_tokenizer = TreebankWordTokenizer()

        self.stopwords = set(stopwords.words('english'))

        self.assertives_hooper1975 = []
        with open('bias_related_lexicons/assertives_hooper1975.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.assertives_hooper1975.append(l)

        self.factives_hooper1975 = []
        with open('bias_related_lexicons/factives_hooper1975.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.factives_hooper1975.append(l)

        self.hedges_hyland2005 = []
        with open('bias_related_lexicons/hedges_hyland2005.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.hedges_hyland2005.append(l)

        self.implicatives_karttunen1971 = []
        with open('bias_related_lexicons/implicatives_karttunen1971.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.implicatives_karttunen1971.append(l)

        self.report_verbs = []
        with open('bias_related_lexicons/report_verbs.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.report_verbs.append(l)

        self.positive_opinion = []
        with open('opinion-lexicon-English/positive-words.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.positive_opinion.append(l)

        self.negative_opinion = []
        with open('opinion-lexicon-English/negative-words.txt', encoding='ISO-8859-1') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.negative_opinion.append(l)

        self.bias_lexicon = []
        with open('bias-lexicon/bias-lexicon.txt') as bl:
            for l in bl.readlines():
                if not l[0]=='#' and len(l)>0: self.bias_lexicon.append(l)

        self.wna        = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')
        self.afinn      = Afinn()
        self.lemmatizer = WordNetLemmatizer()

        self.nrc_lexicon_df = pd.read_csv('dictionaries/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', sep='\t')
        self.nrc_lexicon_dict    = {}

        for i in range(self.nrc_lexicon_df.shape[0]):
            row = self.nrc_lexicon_df.iloc[i]

            if int(row['association']) == 0: continue
            emotion = row['emotion']
            word    = row['word']

            if not emotion in self.nrc_lexicon_dict: self.nrc_lexicon_dict[emotion] = []
            self.nrc_lexicon_dict[emotion].append(word)

        self.afinn_nrc = {
            'anger':        [self.afinn.score(w) for w in self.nrc_lexicon_dict['anger']],
            'fear':         [self.afinn.score(w) for w in self.nrc_lexicon_dict['fear']],
            'disgust':      [self.afinn.score(w) for w in self.nrc_lexicon_dict['disgust']],
            'trust':        [self.afinn.score(w) for w in self.nrc_lexicon_dict['trust']],
            'anticipation': [self.afinn.score(w) for w in self.nrc_lexicon_dict['anticipation']],
            'surprise':     [self.afinn.score(w) for w in self.nrc_lexicon_dict['surprise']],
            'joy':          [self.afinn.score(w) for w in self.nrc_lexicon_dict['joy']],
            'sadness':      [self.afinn.score(w) for w in self.nrc_lexicon_dict['sadness']],
        }

    def get_emotion(self, text):
        tagged = self.treebank_pos_tag(text)
        for t in tagged:
            emotion = self.wna.get_emotion(t[0].lower(), t[1])
            yield emotion

    def calculate_nrc(self, text):
        treebanked = self.treebank_pos_tag(text)
        emotion_dict = {
            'anger':        {'score': 0, 'words': []},
            'fear':         {'score': 0, 'words': []},
            'anticipation': {'score': 0, 'words': []},
            'trust':        {'score': 0, 'words': []},
            'surprise':     {'score': 0, 'words': []},
            'sadness':      {'score': 0, 'words': []},
            'joy':          {'score': 0, 'words': []},
            'disgust':      {'score': 0, 'words': []}
        }

        for word in treebanked:
            word = word[0]

            for emotion in emotion_dict.keys():
                if word in self.nrc_lexicon_dict[emotion]:
                    emotion_dict[emotion]['score'] += 1
                    emotion_dict[emotion]['words'].append(word)

        for emotion in emotion_dict.keys(): emotion_dict[emotion]['average_score'] = float(emotion_dict[emotion]['score']) / float(len(treebanked))

        return emotion_dict

    def get_wordnet_pos(treebank_tag):

        if   treebank_tag.startswith('J'): return wordnet.ADJ
        elif treebank_tag.startswith('V'): return wordnet.VERB
        elif treebank_tag.startswith('N'): return wordnet.NOUN
        elif treebank_tag.startswith('R'): return wordnet.ADV
        else:                              return wordnet.NOUN

    def count_lex_entries(self, text, lexicon):
        count = []
        if not isinstance(text, str): text = ""
        for w in self.treebank_tokenize(text):
            if w in lexicon: count.append(w)

        return (len(count), set(count))

    def extract_bias(self, text):
        bias = {}
        if not isinstance(text, str): text = ""
        bias['assertives_hooper1975']      = self.count_lex_entries(text, self.assertives_hooper1975)[0]
        bias['factives_hooper1975']        = self.count_lex_entries(text, self.factives_hooper1975)[0]
        bias['hedges_hyland2005']          = self.count_lex_entries(text, self.hedges_hyland2005)[0]
        bias['implicatives_karttunen1971'] = self.count_lex_entries(text, self.implicatives_karttunen1971)[0]
        bias['report_verbs']               = self.count_lex_entries(text, self.report_verbs)[0]
        bias['positive_opinion']           = self.count_lex_entries(text, self.positive_opinion)[0]
        bias['negative_opinion']           = self.count_lex_entries(text, self.negative_opinion)[0]
        bias['bias_lexicon']               = self.count_lex_entries(text, self.bias_lexicon)[0]

        return bias

    def sentence_tokenizer(self, text):

        sentence_list = text.split('\n')

        sentence_list = list(itertools.chain.from_iterable([sent_tokenize(s) for s in sentence_list]))

        return sentence_list

    def blli_tag_text(self, text):
        tags = []
        for t in self.rrp.tag(text):
            tags.append({"token": t[0], "tag": t[1]})

        return {"tags": tags}

    def blli_parse_text(self, text):
        if len(text.strip()) == 0: parsed_text = '(S1)'
        else: parsed_text = self.rrp.simple_parse(self.treebank_tokenize(text))

        return {"parse": parsed_text}

    def extract_phrases(self, tree, phrase):
        phrases = []
        depths = []

        if (tree.label() == phrase):
            phrases.append(tree.copy(True))

        for child in tree:
            if (type(child) is Tree):
                list_of_phrases = self.extract_phrases(child, phrase)
                if (len(list_of_phrases) > 0):
                    phrases.extend(list_of_phrases)

        return phrases

    def tree_depth(self, tree):
        n_leaves = len(tree.leaves())
        leavepos = set(tree.leaf_treeposition(n) for n in range(n_leaves))

        depth_dic = {}
        for pos in tree.treepositions():
            if pos not in leavepos:
                if tree[pos].label() in depth_dic:
                    depth_dic[tree[pos].label()] = max(depth_dic[tree[pos].label()], len(pos))
                else:
                    depth_dic[tree[pos].label()] = len(pos)

        return depth_dic

    def stopword_sequence(self, text):
        tokens       = self.treebank_tokenize(text)
        stopword_seq = []

        for token in tokens:
            if token in stopwords: stopword_seq.append(token)

        return ' '.join(stopword_seq)

    def treebank_tokenize(self, text): return self.treebank_tokenizer.tokenize(text)

    def treebank_pos_tag(self, text): return nltk.pos_tag(self.treebank_tokenize(text))

    def treebank_pos_sequence(self, text): return ' '.join([tag[1] for tag in nltk.pos_tag(self.treebank_tokenize(text))])

    def sentence_pos_sequence(self, text): return [self.treebank_pos_sequence(sentence) for sentence in sent_tokenize(text)]

    def extract_liwc(self, text):

        liwc_features = {}
        for liwc_category in self.liwc_categories: liwc_features[liwc_category] = 0

        liwc_features.update(self.liwc.process_text(text))
        liwc_features = {'liwc_' + k: v for k, v in liwc_features.items()}

        return liwc_features

    def _sentence_complexity(self, sentence):
        sentence = sentence.lower()

        if len(self.treebank_tokenize(sentence)) > 70 and len(self.sentence_tokenizer(sentence)) > 1:
            _multi_sents = numpy.array([list(self._sentence_complexity(sent).values()) for sent in self.sentence_tokenizer(sentence)])
            _multi_sents = [s for s in _multi_sents if not any(elem is None for elem in s)]
            if len(_multi_sents) == 0: _multi_sents = numpy.array([[0,0,0,0,0]])

            _multi_sents = numpy.mean(_multi_sents, axis=0)
            return {
                'np_count': _multi_sents[0],
                'vp_count': _multi_sents[1],
                'syntax_depth': _multi_sents[2],
                'vp_depth': _multi_sents[3],
                'np_depth': _multi_sents[4]
            }

        elif len(self.treebank_tokenize(sentence)) > 70:

            return {
                'np_count': None,
                'vp_count': None,
                'syntax_depth': None,
                'vp_depth': None,
                'np_depth': None
            }

        if not sentence.translate(str.maketrans('', '', string.punctuation)).strip(): return {
            'np_count': 0,
            'vp_count': 0,
            'syntax_depth': 0,
            'vp_depth': 0,
            'np_depth': 0
        }
        tree_str = self.blli_parse_text(sentence)['parse']

        blli_tree = Tree.fromstring(tree_str)

        s_np_count = len(self.extract_phrases(blli_tree, 'NP'))
        s_vp_count = len(self.extract_phrases(blli_tree, 'VP'))

        depth_dic = self.tree_depth(blli_tree)
        s_depth   = max(list(depth_dic.values()))
        np_depth  = depth_dic['NP'] if 'NP' in depth_dic.keys() else 0
        vp_depth  = depth_dic['VP'] if 'VP' in depth_dic.keys() else 0

        return {
            'np_count':     s_np_count,
            'vp_count':     s_vp_count,
            'syntax_depth': s_depth,
            'vp_depth':     vp_depth,
            'np_depth':     np_depth
        }

    def syntax_complexity_analysis(self, sentences):
        sentence_complexities = [list(self._sentence_complexity(sentence).values()) for sentence in sentences]

        sentence_complexities = [s for s in sentence_complexities if not any(elem is None for elem in s)]

        if len(sentence_complexities) < 1:
            return {
                'quantile_25': 0,
                'quantile_75': 0,
                'median': 0,
                'mean': 0,
                'max': 0,
                'min': 0,
                'std': 0
            }

        sentence_complexities = numpy.array(sentence_complexities)

        q25     = numpy.percentile(sentence_complexities, 25, axis=0)
        q75     = numpy.percentile(sentence_complexities, 75, axis=0)
        medians = numpy.median(sentence_complexities, axis=0)
        means   = numpy.mean(sentence_complexities, axis=0)
        maxes   = numpy.max(sentence_complexities, axis=0)
        mins    = numpy.min(sentence_complexities, axis=0)
        stds    = numpy.std(sentence_complexities, axis=0)

        return {
            'quantile_25': q25,
            'quantile_75': q75,
            'median':      medians,
            'mean':        means,
            'max':         maxes,
            'min':         mins,
            'std':         stds
        }

    def _extract_features(self, text):
        features = {}

        features.update(surface_features(text))
        features.update(psychological_features(text))
        features.update(pos_features(text))
        features.update(readability_index(text))
        features.update(vocabulary_richness(text))
        features.update(sentiment(text))
        features.update(self.extract_liwc(text))
        features.update(self.extract_bias(text))

        return features

    def explode_complexity(self, complexity_analysis_entry):

        complexity_column_names = ['np_count', 'vp_count', 'syntax_depth', 'vp_depth', 'np_depth']
        ca_dict                 = {}

        for k,v in complexity_analysis_entry.items():
            for i in range(len(complexity_column_names)):
                ccn = complexity_column_names[i]
                ca_dict[ccn + '_' + k] = v[i]

        return ca_dict

    def extract_features(self):

        self.df_content_liwc = pd.DataFrame([self.extract_liwc(content) for content in self.df['text']])

        self.df_content_pos_sequences = self.df['text'].apply(self.treebank_pos_sequence)

        content_word_vectorizer = TfidfVectorizer(analyzer='word', min_df=5, max_df=.8, ngram_range=(1,3))
        content_word_vectorizer.fit(self.df_content_pos_sequences)
        self.content_pos_ngrams_array = content_word_vectorizer.transform(self.df_content_pos_sequences)

        self.content_pos_ngrams_features  = pd.DataFrame(self.content_pos_ngrams_array.todense(), columns=content_word_vectorizer.get_feature_names())
        self.content_pos_ngrams_features  = self.content_pos_ngrams_features.add_prefix('pos_ngrams_')

        self.df_content_stopword_sequences = self.df['text'].apply(self.stopword_sequence)

        self.df_content_stopword_sequences = self.df_content_stopword_sequences.replace(numpy.nan, '', regex=True)

        self.content_word_vectorizer = TfidfVectorizer(analyzer='word', min_df=5, max_df=.8, ngram_range=(1,3))
        self.content_word_vectorizer.fit(self.df_content_stopword_sequences)
        self.content_stopword_ngrams_array = content_word_vectorizer.transform(self.df_content_stopword_sequences)

        self.content_stopword_ngrams_features = pd.DataFrame(
            self.content_stopword_ngrams_array.todense(),
            columns=self.content_word_vectorizer.get_feature_names()
        )

        self.content_stopword_ngrams_features = self.content_stopword_ngrams_features.add_prefix('stopword_ngrams_')
        self.df_content_punctuation_sequences = self.df['text'].apply(punctuation_sequence)
        self.df_content_punctuation_sequences = self.df_content_punctuation_sequences.replace(numpy.nan, '', regex=True)

        self.content_punctuation_vectorizer = TfidfVectorizer(analyzer='char', min_df=5, max_df=.8, ngram_range=(1,4))
        self.content_punctuation_vectorizer.fit(self.df_content_punctuation_sequences)
        self.content_punctuation_ngram_array = self.content_punctuation_vectorizer.transform(self.df_content_punctuation_sequences)

        self.content_punctuation_ngram_features = pd.DataFrame(self.content_punctuation_ngram_array.todense(), columns=self.content_punctuation_vectorizer.get_feature_names())
        self.content_punctuation_ngram_features = self.content_punctuation_ngram_features.add_prefix('punctuation_ngrams_')

        self.content_sentences = self.df['text'].apply(self.sentence_tokenizer)

        self.content_bias = self.df['text'].apply(self.extract_bias)
        self.content_bias_df = pd.DataFrame.from_dict(list(self.content_bias.values))

        self.df['text_len'] = self.df['text'].apply(len)

        self.content_linguistic_features = self.df['text'].apply(self._extract_features)
        f_dict = self.content_linguistic_features.T.to_dict()
        f_dict = list(f_dict.values())

        self.content_linguistic_features_df = pd.DataFrame.from_dict(list(self.content_linguistic_features.values))

        return self.content_linguistic_features_df