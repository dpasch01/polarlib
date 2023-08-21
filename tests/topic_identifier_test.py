import unittest
from unittest.mock import MagicMock, patch
from polarlib.topic_identifier import TopicIdentifier

class TestTopicIdentifier(unittest.TestCase):

    @patch("polarlib.topic_identifier.load_article")
    @patch("polarlib.topic_identifier.sentence_transformers.SentenceTransformer")
    @patch("polarlib.topic_identifier.stopwords.words")
    @patch("polarlib.topic_identifier.tqdm")
    def setUp(self, mock_tqdm, mock_stopwords, mock_sentence_transformer, mock_load_article):
        self.mock_load_article = mock_load_article
        self.mock_load_article.return_value = {
            "noun_phrases": [{"noun_phrases": [{"ngram": "example noun phrase"}]}]
        }
        self.mock_stopwords = mock_stopwords
        self.mock_stopwords.return_value = ["english", "stopwords"]
        self.mock_sentence_transformer = mock_sentence_transformer
        self.mock_transformer_instance = MagicMock()
        self.mock_sentence_transformer.return_value = self.mock_transformer_instance
        self.mock_tqdm = mock_tqdm
        self.mock_tqdm.side_effect = lambda x: x

    def test_init(self):
        topic_identifier = TopicIdentifier(output_dir="./example")

        self.assertEqual(topic_identifier.hyphen_regex, r'(?=\S+[-])([a-zA-Z-]+)')
        self.assertEqual(topic_identifier.output_dir, "./example")
        self.assertEqual(topic_identifier.noun_phrase_path_list, [])
        self.assertEqual(topic_identifier.english_stopwords, ["english", "stopwords"])
        self.assertIsNone(topic_identifier.clean_noun_phrase_list)
        self.assertIsNone(topic_identifier.encoded_noun_phrase_list)
        self.assertEqual(topic_identifier.noun_phrase_embedding_dict, {})
        self.assertEqual(topic_identifier.model, self.mock_transformer_instance)
        self.assertEqual(len(self.mock_sentence_transformer.call_args_list), 1)
        self.assertEqual(len(self.mock_load_article.call_args_list), 0)

    @patch("polarlib.topic_identifier.TextBlob")
    def test_lemmatize(self, mock_text_blob):
        mock_blob_instance = MagicMock()
        mock_text_blob.return_value = mock_blob_instance
        mock_blob_instance.tags = [("word", "N"), ("example", "V")]
        topic_identifier = TopicIdentifier(output_dir="./example")

        result = topic_identifier._lemmatize("example noun phrase")

        self.assertEqual(result, "word example")
        self.assertEqual(len(mock_text_blob.call_args_list), 1)

    def test_pipeline_func(self):
        topic_identifier = TopicIdentifier(output_dir="./example")

        result = topic_identifier._pipeline_func("Example Noun Phrase", [str.lower, str.strip])

        self.assertEqual(result, "example noun phrase")

    @patch("polarlib.topic_identifier.nltk.word_tokenize")
    def test_tokenize(self, mock_word_tokenize):
        mock_word_tokenize.return_value = ["example", "noun", "phrase"]
        topic_identifier = TopicIdentifier(output_dir="output_directory")

        result = topic_identifier._tokenize("example noun phrase")

        self.assertEqual(result, ["example", "noun", "phrase"])
        self.assertEqual(len(mock_word_tokenize.call_args_list), 1)

    # Add more test cases for other methods as needed

if __name__ == "__main__": unittest.main()