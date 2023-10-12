from __future__ import division
from textblob import TextBlob

import sys

def sentiment(text):
    
    if not isinstance(text, str): text = ""
    blob = TextBlob(text)

    sent = {}
    sent['s_polarity'] = blob.sentiment.polarity
    sent['s_subjectivity'] = blob.sentiment.subjectivity

    return sent

def main():
	textcontent = sys.argv[1]
	print(sentiment(textcontent))
  
if __name__== "__main__":
	main()
