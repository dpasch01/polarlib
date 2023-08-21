import unittest
from unittest.mock import patch, MagicMock
from polarlib.actor_extractor import EntityExtractor, NounPhraseExtractor

class TestEntityExtractor(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./example"
        self.extractor = EntityExtractor(self.output_dir)

    def test_query_dbpedia_entities_with_mock(self):
        mock_post = MagicMock()
        with patch('requests.post', return_value=mock_post) as mock_requests_post:
            mock_post.json.return_value = {'Resources': [{'URI': 'http://dbpedia.org/resource/Test', 'offset': 10, 'surfaceForm': 'test', 'similarityScore': 0.9, 'percentageOfSecondRank': 0.1, 'types': 'Wikidata:Q123'}]}
            entities = self.extractor.query_dbpedia_entities('Test Text')
            self.assertEqual(len(entities), 1)
            self.assertEqual(entities[0]['title'], 'http://dbpedia.org/resource/Test')

    def test_extract_article_entities_with_mock(self):
        mock_load_article = MagicMock()
        mock_load_article.return_value = {
            'uid': '123',
            'text': 'Sample article text'
        }
        with patch('polarlib.load_article', return_value=mock_load_article):
            result = self.extractor.extract_article_entities('sample_path')
            self.assertTrue(result)

class TestNounPhraseExtractor(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./example"
        self.extractor = NounPhraseExtractor(self.output_dir, spacy_model_str="en_core_web_sm")

    def test_clean_text(self):
        cleaned_text = self.extractor._clean_text("Sample text with punctuation.")
        self.assertEqual(cleaned_text, "sample text punctuation")

    def test_extract_article_noun_phrases_with_mock(self):
        mock_load_article = MagicMock()
        mock_load_article.return_value = {
            'uid': '123',
            'entities': [
                {
                    'sentence': 'Sample sentence with entity.',
                    'from': 0,
                    'to': 32,
                    'entities': [
                        {
                            'begin': 7,
                            'end': 14
                        }
                    ]
                }
            ]
        }
        with patch('polarlib.load_article', return_value=mock_load_article):
            result = self.extractor.extract_article_noun_phrases('sample_path')
            self.assertTrue(result)

if __name__ == '__main__': unittest.main()