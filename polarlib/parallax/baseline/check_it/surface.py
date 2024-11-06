from __future__ import division

from pprint import pprint
import re, sys, nltk, string, collections, json, spacy
from nltk.corpus import stopwords
from nltk import sent_tokenize
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def surface_features(text):
    dict_feature = {}

    sentences = []
    if not isinstance(text, str): text = ""
    if text is None: text = ""
    if not len(text)<=5:
        #print(text)
        #print(len(text))
        doc = nlp(text)
        sentences =  [sent.text.strip() for sent in doc.sents]

    total_number_of_sentences = len(list(sentences))
    if total_number_of_sentences == 0:
        total_number_of_sentences = 1

    total_number_of_words = 0
    total_number_of_characters = 0
    total_number_of_begin_upper = 0
    total_number_of_begin_lower = 0
    total_number_of_all_caps = 0
    total_number_of_stopwords = 0
    total_number_of_lines = len(text.splitlines())

    ratio_alphabetic = 0
    ratio_uppercase = 0
    ratio_digit = 0

    avg_number_of_characters_per_word = 0
    avg_number_of_words_per_sentence = 0
    avg_number_of_characters_per_sentence = 0
    avg_number_of_begin_upper_per_sentence = 0
    avg_number_of_begin_lower_per_sentence = 0
    avg_number_of_all_caps_per_sentence = 0
    avg_number_of_stopwords_per_sentence = 0

    sen_i = 0
    for sentence in sentences:

        sentence = nltk.word_tokenize(sentence)

        original_words = [w.rstrip(string.punctuation).lstrip(string.punctuation) for w in sentence]
        
        remove_words = []
        for i in range(len(original_words)):
            if original_words[i] in string.punctuation:
                remove_words.append(original_words[i])

        for word in remove_words:
            original_words.remove(word)

        r_alphabetic = 0
        r_uppercase = 0
        r_digit = 0

        words_nosw = original_words[:]
        word_length = 0
        all_cap_words = 0
        begin_lower_case = 0
        begin_upper_case = 0
    
        for word in original_words:

            alphabetic = [c.lower() for c in word if c.isalpha()]

            r_alphabetic += len(alphabetic)
            r_uppercase += len([c.lower() for c in word if c.isupper()])
            r_digit += len([c.lower() for c in word if c.isdigit()])
            word_length += len(word)

            if word.isupper():
                all_cap_words += 1
            elif word[0].islower():
                begin_lower_case += 1
            elif word[0].isupper():
                begin_upper_case += 1

            if word in stopwords.words('english'):
                words_nosw.remove(word)

        total_number_of_words += len(original_words)
        total_number_of_characters += word_length
        total_number_of_begin_upper += begin_upper_case
        total_number_of_begin_lower += begin_lower_case
        total_number_of_all_caps += all_cap_words
        total_number_of_stopwords += (len(original_words) - len(words_nosw) + 1)

        ratio_alphabetic += r_alphabetic
        ratio_uppercase += r_uppercase
        ratio_digit += r_digit
                

        avg_number_of_characters_per_word += division(word_length, len(original_words))
        avg_number_of_begin_upper_per_sentence += division(begin_upper_case, len(original_words))
        avg_number_of_begin_lower_per_sentence += division(begin_lower_case, len(original_words))
        avg_number_of_all_caps_per_sentence += division(all_cap_words, len(original_words))
        avg_number_of_stopwords_per_sentence += division((len(original_words) - len(words_nosw) + 1), len(original_words))

    avg_number_of_characters_per_word = division(avg_number_of_characters_per_word, total_number_of_sentences)
    avg_number_of_words_per_sentence = division(total_number_of_words, total_number_of_sentences)
    avg_number_of_characters_per_sentence = division(total_number_of_characters, total_number_of_sentences)
    avg_number_of_begin_upper_per_sentence = division(avg_number_of_begin_upper_per_sentence, total_number_of_sentences)
    avg_number_of_begin_lower_per_sentence = division(avg_number_of_begin_lower_per_sentence, total_number_of_sentences)
    avg_number_of_all_caps_per_sentence = division(avg_number_of_all_caps_per_sentence, total_number_of_sentences)
    avg_number_of_stopwords_per_sentence = division(avg_number_of_stopwords_per_sentence, total_number_of_sentences)

    ratio_alphabetic = division(ratio_alphabetic, total_number_of_characters)
    ratio_uppercase = division(ratio_uppercase, total_number_of_characters)
    ratio_digit = division(ratio_digit, total_number_of_characters)

    dict_feature['total_number_of_sentences'] = total_number_of_sentences
    dict_feature['total_number_of_words'] = total_number_of_words
    dict_feature['total_number_of_characters'] = total_number_of_characters
    dict_feature['total_number_of_begin_upper'] = total_number_of_begin_upper
    dict_feature['total_number_of_begin_lower'] = total_number_of_begin_lower
    dict_feature['total_number_of_all_caps'] = total_number_of_all_caps
    dict_feature['total_number_of_stopwords'] = total_number_of_stopwords
    dict_feature['total_number_of_lines'] = total_number_of_lines

    dict_feature['ratio_alphabetic'] = ratio_alphabetic
    dict_feature['ratio_uppercase'] = ratio_uppercase
    dict_feature['ratio_digit'] = ratio_digit

    dict_feature['avg_number_of_characters_per_word'] = avg_number_of_characters_per_word
    dict_feature['avg_number_of_words_per_sentence'] = avg_number_of_words_per_sentence
    dict_feature['avg_number_of_characters_per_sentence'] = avg_number_of_characters_per_sentence
    dict_feature['avg_number_of_begin_upper_per_sentence'] = avg_number_of_begin_upper_per_sentence
    dict_feature['avg_number_of_all_caps_per_sentence'] = avg_number_of_all_caps_per_sentence
    dict_feature['avg_number_of_begin_lower_per_sentence'] = avg_number_of_begin_lower_per_sentence
    dict_feature['avg_number_of_stopwords_per_sentence'] = avg_number_of_stopwords_per_sentence

    dict_feature = {'sf_' + k: v for k, v in dict_feature.items()}

    return dict_feature

def division(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0

def main():
    textcontent = sys.argv[1]
    pprint(surface_features(textcontent))

if __name__ == "__main__":
    main()
