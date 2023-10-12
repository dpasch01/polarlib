from __future__ import division

import sys, string
from pprint import pprint
from collections import Counter
from string import punctuation

d_punctuation = dict.fromkeys(list(punctuation), 0)

def punctuation_features(text):
    c_punctuation = Counter(c for c in text if c in d_punctuation)
    d_punctuation.update(dict(c_punctuation))
    d_punctuation['n_of_punctuations'] = sum(c_punctuation.values())

    return {'pnc_' + k: v for k, v in d_punctuation.items()}

def main():
    textcontent=sys.argv[1]
    pprint(punctuation_features(textcontent))

if __name__== "__main__":
    main()
