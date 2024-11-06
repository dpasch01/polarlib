from __future__ import division
from collections import Counter
import nltk, sys, math
import numpy as np
from nltk.tokenize import WhitespaceTokenizer

POS_TAGS = ["$", "``", "''", "(", ")", ",", "--", ".", ":","#", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
POS_TAGS_TRANSLATION = {
	"''": "n_of_single_quotes",
	"``": "n_of_double_quotes",
	"CC": "n_of_conjunctions",
	"CD": "n_of_numerals_cardinals",
	"DT": "n_of_determiners",
	"EX": "n_of_existentials",
	"FW": "n_of_foreign_words",
	"IN": "n_of_prepositions",
	"JJ": "n_of_numeral_adjectives",
	"JJR": "n_of_comparative_adjectives",
	"JJS": "n_of_superlative_adjectives",
	"LS": "n_of_lists",
	"MD": "n_of_auxiliary_modals",
	"NN": "n_of_nouns",
	"NNP": "n_of_proper_nouns",
	"NNPS": "n_of_proper_nouns_plural",
	"NNS": "n_of_nouns_plural",
	"PDT": "n_of_pre_determinents",
	"POS": "n_of_genitive_markers",
	"PRP": "n_of_pronouns",
	"PRP$": "n_of_possesive_pronouns",
	"RB": "n_of_adverbs",
	"RBR": "n_of_comparative_adverbs",
	"RBS": "n_of_superlative_adverbs",
	"RP": "n_of_particles",
	"SYM": "n_of_symbols",
	"TO": "n_of_to_prepositions",
	"UH": "n_of_interjection",
	"VB": "n_of_base_verbs",
	"VBD": "n_of_past_verbs",
	"VBG": "n_of_present_verbs",
	"VBN": "n_of_past_particle_verbs",
	"VBP": "n_of_present_not_3rd_verbs",
	"VBZ": "n_of_present_3rd_verbs",
	"WDT": "n_of_wh_determiners",
	"WP": "n_of_wh_pronouns",
	"WP$": "n_of_wh_pronouns_possesive",
	"WRB": "n_of_wh_adverbs"
}

def pos_features(text):
    
	if not isinstance(text, str): text = ""    
	features = dict.fromkeys(POS_TAGS, 0)
	text = nltk.word_tokenize(text)

	pos = nltk.pos_tag(text)

	pos_array = np.asarray(pos)
	
	

	n_of_i_pronoun = 0
	n_of_we_pronoun = 0
	n_of_you_pronoun = 0
	n_of_he_she_pronoun = 0
	n_of_quotes = 0
	n_of_quoted_content = 0

	counter = dict(Counter([j for i,j in pos]))
	
	for p in pos:
		if p[1] == "''" or p[1] == "``" :
			n_of_quotes = n_of_quotes + 1
		
		if p[1] == 'PRP':
			if p[0].lower() == 'i': n_of_i_pronoun = n_of_i_pronoun + 1
			if p[0].lower() == 'we': n_of_we_pronoun = n_of_we_pronoun + 1
			if p[0].lower() == 'you': n_of_you_pronoun = n_of_you_pronoun + 1
			if p[0].lower() == 'he' or p[0].lower() == 'she' : n_of_he_she_pronoun = n_of_he_she_pronoun + 1


	n_of_quoted_content = math.ceil(float(n_of_quotes) / 2)

	for key in counter.keys():
		features[key] = counter[key]

	features['n_of_future_tense_words'] = features['MD']
	features['n_of_past_tense_words'] = features['VBN'] + features['VBD']
	features['n_of_present_tense_words'] = features['VBZ'] + features['VBG'] + features['VBP']
	features['n_of_i_pronoun'] = n_of_i_pronoun 
	features['n_of_we_pronoun'] = n_of_we_pronoun
	features['n_of_you_pronoun'] = n_of_you_pronoun
	features['n_of_he_she_pronoun'] = n_of_he_she_pronoun 

	features = {'pos_' + k: v for k, v in features.items()}

	features['sf_n_of_quotes'] = n_of_quotes 
	features['sf_n_of_quoted_content'] = n_of_quoted_content 
	return features

def main():
	textcontent = sys.argv[1]
	print(pos_features(textcontent))

if __name__== "__main__":
	main()
