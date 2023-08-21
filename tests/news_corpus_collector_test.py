import unittest, os
from datetime import date
from polarlib.news_corpus_collector import URLKeywordExtractor, NewsCorpusCollector

class TestURLKeywordExtractor(unittest.TestCase):

    def setUp(self):
        self.url_list = [
            'https://www.businessinsider.com/everything-you-need-to-know-about-chat-gpt-2023-1',
        ]
        self.extractor = URLKeywordExtractor(self.url_list)

    def test_extract_keywords(self):
        keywords = self.extractor.extract_keywords(n=10)
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) <= 10)

class TestNewsCorpusCollector(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./example"
        self.keywords = ["openai", "gpt"]
        self.collector = NewsCorpusCollector(
            output_dir=self.output_dir,
            from_date=date(year=2023, month=8, day=15),
            to_date=date(year=2023, month=8, day=15),
            keywords=self.keywords
        )

    def test_collect_archives(self):
        print('Fetching News Archives')
        self.collector.collect_archives()
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'dumps')))

    def test_collect_articles(self):
        print('Collecting Articles')
        self.collector.collect_articles()

    def test_pre_process_articles(self):
        print('Pre-Process Articles')
        self.collector.pre_process_articles()

if __name__ == "__main__": unittest.main()