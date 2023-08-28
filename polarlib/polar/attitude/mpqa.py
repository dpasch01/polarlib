import pandas as pd
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

from polarlib.utils.utils import *

class mpqa:

    """
    A class representing the MPQA sentiment analysis model.

    This class provides functionality for loading the MPQA lexicon, converting part-of-speech tags,
    and calculating sentiment scores using the MPQA lexicon.

    Args:
        path (str): Path to the MPQA lexicon data.

    Attributes:
        mpqa_path (str): Path to the MPQA lexicon data.
        MPQA (dict): A dictionary containing MPQA lexicon data.

    Methods:
        load_mpqa(): Load the MPQA lexicon data.
        convert_to_mpqa_pos(pos): Convert part-of-speech tag to MPQA compatible format.
        calculate_mpqa(tokens): Calculate sentiment scores using the MPQA lexicon.
    """

    def __init__(self, path):
        """
        Initialize the mpqa class.

        Args:
            path (str): Path to the MPQA lexicon data.
        """
        self.mpqa_path = path
        self.MPQA      = None

    def load_mpqa(self):
        """
        Load the MPQA lexicon data.
        """
        mpqa_df = []

        with open(self.mpqa_path, 'r') as f:
            for l in f.readlines():
                obj = {}

                for d in l.strip().split(' '):
                    d = d.split('=')
                    if len(d) < 2: continue
                    obj[d[0]] = d[1] if d[0] != 'len' else int(d[1])

                mpqa_df.append(obj)

        mpqa_df = pd.DataFrame.from_dict(mpqa_df).set_index('word1')

        self.MPQA = mpqa_df.T.to_dict()

    def convert_to_mpqa_pos(self, pos):
        """
        Convert a part-of-speech tag to MPQA compatible format.

        Args:
            pos (str): Part-of-speech tag to be converted.

        Returns:
            str: MPQA compatible part-of-speech.
        """
        if    pos == 'VERB': return 'verb'
        elif  pos == 'NOUN' or pos == 'PROPN': return 'noun'
        elif  pos == 'ADJ': return 'adj'
        elif  pos == 'ADV': return 'adverb'
        else: return 'other'

    def calculate_mpqa(self, tokens):
        """
        Calculate sentiment scores using the MPQA lexicon.

        Args:
            tokens (list): List of tokens for sentiment analysis.

        Returns:
            float: Calculated sentiment score.
        """
        positive_list, negative_list = [], []
        positive_words, negative_words = [], []

        for token in tokens:

            t = token.text.lower().strip()
            if t in stop_words: continue
            if not t in list(self.MPQA.keys()): t = token.lemma_.lower().strip()
            if not t in list(self.MPQA.keys()): continue
            if t in stop_words: continue

            if 'debate'  == t: continue
            if 'victory' == t: continue
            if 'defeat'  == t: continue
            if 'force'   == t: continue

            mpqa_obj = self.MPQA[t]

            """
            t_pos = self.convert_to_mpqa_pos(token.tag)

            if not (mpqa_obj['pos1'] == 'anypos' or t_pos == mpqa_obj['pos1']): continue
            """

            mpqa_polarity = mpqa_obj['priorpolarity']

            if mpqa_polarity == 'positive' or mpqa_polarity == 'both':
                positive_words.append(t)
                positive_list.append(1.0)

            if mpqa_polarity == 'negative' or mpqa_polarity == 'both':
                negative_words.append(t)
                negative_list.append(1.0)

        if len(positive_list + negative_list) == 0: return 0.00

        return sentiment_threshold_difference(
            sum(positive_list) / len(positive_list + negative_list),
            abs(sum(negative_list)) / len(positive_list + negative_list)
        )

        """
        return {'POSITIVE': sum(positive_list), 'NEGATIVE': abs(sum(negative_list))}, \
            {'POSITIVE': positive_words, 'NEGATIVE': negative_words}
        """