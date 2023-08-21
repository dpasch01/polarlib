import unittest
import os
from polarlib.actor_extractor import EntityExtractor, NounPhraseExtractor

class TestEntityExtractor(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./example"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def test_extract_entities(self):
        entity_extractor = EntityExtractor(output_dir=self.output_dir)
        entity_extractor.extract_entities()

        # Add assertions here to verify the correctness of the extraction

class TestNounPhraseExtractor(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./example"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def test_extract_noun_phrases(self):
        noun_phrase_extractor = NounPhraseExtractor(output_dir=self.output_dir)
        noun_phrase_extractor.extract_noun_phrases()

        # Add assertions here to verify the correctness of the extraction

if __name__ == "__main__": unittest.main()