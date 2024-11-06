from __future__ import division
from textstat.textstat import textstat
import sys

def readability_index(text):
    
	if not isinstance(text, str): text = ""
	readability_index = {}

	readability_index['ri_no_of_syllables'] = textstat.syllable_count(text, lang='en_US')
	readability_index["ri_flesch"] = textstat.flesch_reading_ease(text)
	readability_index["ri_smog"] = textstat.smog_index(text)
	readability_index["ri_flesch_kincaid"] = textstat.flesch_kincaid_grade(text)
	readability_index["ri_coleman_liau"] = textstat.coleman_liau_index(text)
	readability_index["ri_ari"] = textstat.automated_readability_index(text)
	readability_index["ri_dale_chall"] = textstat.dale_chall_readability_score(text)
	readability_index["ri_difficult_words"] = textstat.difficult_words(text)
	readability_index["ri_linsear"] = textstat.linsear_write_formula(text)
	readability_index["ri_gunning_fog"] = textstat.gunning_fog(text)
		
	return readability_index 

def main():
	textcontent = sys.argv[1]
	print(readability_index(textcontent))
  
if __name__== "__main__":
	main()
