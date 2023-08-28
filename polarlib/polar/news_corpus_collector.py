import json, itertools, multiprocessing, time, os, wget, urllib, numpy, string, re, zipfile, pandas as pd, sys

from tqdm import tqdm
from keybert import KeyBERT
from newspaper import Article, Config
from datetime import datetime, date, timedelta
from multiprocessing import Pool

GDELT_BASE = 'http://data.gdeltproject.org/events/{}.export.CSV.zip'

GDELT_FIELDS = [
    'globaleventid', 'day', 'monthyear', 'year', 'fractiondate', 'actor1code', 'actor1name', 'actor1countrycode',
    'actor1knowngroupcode', 'actor1ethniccode', 'actor1religion1code', 'actor1religion2code', 'actor1type1code',
    'actor1type2code', 'actor1type3code', 'actor2code', 'actor2name', 'actor2countrycode', 'actor2knowngroupcode',
    'actor2ethniccode', 'actor2religion1code', 'actor2religion2code', 'actor2type1code', 'actor2type2code',
    'actor2type3code', 'isrootevent', 'eventcode', 'eventbasecode', 'eventrootcode', 'quadclass', 'goldsteinscale',
    'nummentions', 'numsources', 'numarticles', 'avgtone', 'actor1geo_type', 'actor1geo_fullname',
    'actor1geo_countrycode', 'actor1geo_adm1code', 'actor1geo_lat', 'actor1geo_long', 'actor1geo_featureid',
    'actor2geo_type', 'actor2geo_fullname', 'actor2geo_countrycode', 'actor2geo_adm1code string', 'actor2geo_lat',
    'actor2geo_long', 'actor2geo_featureid', 'actiongeo_type', 'actiongeo_fullname', 'actiongeo_countrycode',
    'actiongeo_adm1code', 'actiongeo_lat', 'actiongeo_long', 'actiongeo_featureid', 'dateadded', 'sourceurl'
]


class URLKeywordExtractor:

    def __init__(self, url_list):
        """
        Initialize the URLKeywordExtractor instance.

        :param url_list: A list of URLs to extract keywords from.
        """
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.url_list = url_list
        self.model = KeyBERT()

        self.texts = []

        for url in tqdm(self.url_list):
            article = Article(url)

            article.download()
            article.parse()

            self.texts.append(article.text)

    def extract_keywords(self, n=20):
        """
        Extract keywords from the stored article texts.

        :param n: Number of top keywords to extract (default is 20).
        :return: A list of extracted keywords.
        """
        keyword_list = []

        for text in self.texts: keyword_list += [k for k in self.model.extract_keywords(text, top_n=n)]

        keyword_list = sorted(keyword_list, key=lambda t: t[1], reverse=True)[:n]

        return [k[0] for k in keyword_list]


class NewsCorpusCollector:

    def __init__(self, output_dir, from_date, to_date, keywords):
        """
        Initialize the NewsCorpusGenerator.

        :param output_dir: Output directory for storing downloaded data.
        :param from_date: Starting date for data collection.
        :param to_date: Ending date for data collection.
        :param keywords: List of keywords for filtering articles.
        """
        self.output_dir = output_dir
        self.from_date = from_date
        self.to_date = to_date
        self.duration = self.to_date - self.from_date
        self.duration = self.duration.days + 1
        self.keywords = keywords

        if os.path.isdir(self.output_dir):
            print('Warning: Path \'%s\' already exists.' % self.output_dir)
        else:
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, 'dumps'))
            os.makedirs(os.path.join(self.output_dir, 'html'))
            os.makedirs(os.path.join(self.output_dir, 'articles'))

        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        self.config.request_timeout = 3

    def collect_archives(self):
        """
        Collect GDELT archives and store them in the specified output directory.
        """
        for i in tqdm(range(self.duration), desc='Retrieving GDELT Dumps'):
            d = self.from_date + timedelta(days=i)
            d_str = d.strftime('%Y%m%d')

            if os.path.exists(os.path.join(self.output_dir, 'dumps/{}.export.CSV.zip'.format(d_str))): continue

            wget.download(GDELT_BASE.format(d_str), out=os.path.join(self.output_dir, 'dumps'))

        return

    def _get_link_source(self, sourceurl):
        """
        Extract the source URL from a given URL.

        :param sourceurl: Source URL.
        :return: Extracted source.
        """
        if not isinstance(sourceurl, str) and numpy.isnan(sourceurl): return ''
        return urllib.parse.urlparse(sourceurl).netloc

    def _get_link_source_path(self, sourceurl):
        """
        Extract the source URL path from a given URL.

        :param sourceurl: Source URL.
        :return: Extracted source path.
        """
        if not isinstance(sourceurl, str) and numpy.isnan(sourceurl): return ''
        return urllib.parse.urlparse(sourceurl).path

    def _get_query_url(self, url, gd_day, archive_flag=True):
        """
        Generate a query URL for archiving or retrieving.

        :param url: Original URL.
        :param gd_day: GDELT day.
        :param archive_flag: Flag to determine if archiving is used.
        :return: Query URL.
        """
        if archive_flag:
            return 'https://web.archive.org/web/' + str(gd_day) + '00000/' + url
        else:
            return url

    def _format_title(self, s):
        """
        Format a string for use as a title.

        :param s: Input string.
        :return: Formatted string.
        """
        for st in string.punctuation: s = s.replace(st, ' ')
        s = re.sub(' +', '-', s)
        s = s.lower()

        return s[:]

    def retrieve_article(self, article_url, parse_flag=True, nlp_flag=False):
        """
        Retrieve and parse an article from a given URL.

        :param article_url: Article URL.
        :param parse_flag: Flag to indicate parsing.
        :param nlp_flag: Flag to indicate NLP processing.
        :return: Parsed article object.
        """
        article = Article(article_url, config=self.config)
        article.download()

        if parse_flag: article.parse()
        if parse_flag and nlp_flag: article.nlp()

        return article

    def pre_process_article(self, path):
        """
        Pre-process and store the article for later analysis.

        :param path: path to the article .json file
        :return: Boolean
        """
        with open(path, 'r') as f: article_obj = json.load(f)

        article_dict_str = json.dumps({
            'uid': article_obj['uid'],
            'text': self._pipeline_func(
                article_obj['text'],
                [
                    self._replace_special,
                    self._uncontract,
                    lambda t: t.replace('\n', ' ')
                ]
            )
        })

        output_folder = os.path.join(self.output_dir, 'pre_processed/' + path.split('/')[-2] + '/')
        output_file = output_folder + article_obj['uid'] + '.json'

        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
        with open(output_file, 'w') as f:     json.dump(article_dict_str, f)

        return True

    def _replace_special(self, text):
        text = text.replace('``', "''")
        text = text.replace('`', "'")
        text = text.replace('“', '"')
        text = text.replace('”', '"')
        text = text.replace('’', "'")
        text = text.replace('‘', "'")
        text = text.replace("'", "'")
        text = text.replace('–', "-")
        text = text.replace('—', "-")
        text = text.replace('\"', '"')
        text = text.replace("\'", "'")

        return text

    def _uncontract(self, text):
        text = re.sub(
            r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
            r"\1\2 not", text)
        text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
        text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

        text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
        text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
        text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

        return text

    def _pipeline_func(self, text, func_list):
        for f in func_list: text = f(text)
        return text

    def pre_process_articles(self):
        article_paths = [os.path.join(self.output_dir, 'articles/' + p + '/') for p in
                         sorted(os.listdir(os.path.join(self.output_dir, 'articles/')))]
        article_paths = list(itertools.chain.from_iterable([[p + _ for _ in os.listdir(p)] for p in article_paths]))

        pool = Pool(multiprocessing.cpu_count())

        for i in tqdm(
                pool.imap_unordered(self.pre_process_article, article_paths),
                desc='Article Pre-processing',
                total=len(article_paths)
        ): pass

        pool.close()
        pool.join()

    def article_collection_process(self, idxhgd):

        idx, hgd = idxhgd

        archive_url = self._get_query_url(
            hgd['sourceurl'],
            hgd['day'],
            archive_flag=False
        )

        hgd['config_day'] = hgd['d_str']

        try:
            article_obj = self.retrieve_article(archive_url, parse_flag=False)
        except Exception as ex:
            print(idx, archive_url, ex)
            return

        output_folder = os.path.join(self.output_dir, 'html/' + str(hgd['config_day']))
        output_file = os.path.join(output_folder,
                                   hgd['source'] + '.' + self._format_title(self._get_link_source_path(archive_url))[
                                                         :100] + '.html')

        os.makedirs(output_folder, exist_ok=True)
        with open(output_file, 'w') as html_file:
            html_file.write(article_obj.html)

        return True

    def collect_articles(
            self,
            actor1countrycode='USA',
            actor2countrycode='USA',
            n_articles=1000
    ):
        """
        Collect articles from GDELT archives and store them.

        :param actor1countrycode: Actor 1 country code.
        :param actor2countrycode: Actor 2 country code.
        :param n_threads: Number of threads for parallel processing.
        :param use_web_archive: Flag to use web archiving.

        Args:
            n_articles:
        """

        for i in range(self.duration):

            d = self.from_date + timedelta(days=i)
            d_str = d.strftime('%Y%m%d')
            zf = zipfile.ZipFile('{}{}.export.CSV.zip'.format(os.path.join(self.output_dir, 'dumps/'), d_str))

            gd_df = pd.read_csv(zf.open('{}.export.CSV'.format(d_str)), sep='\t', header=None)
            gd_df.columns = GDELT_FIELDS

            gd_df['sourceurl_path'] = gd_df['sourceurl'].apply(self._get_link_source_path)
            gd_df['source'] = gd_df['sourceurl'].apply(self._get_link_source)

            #######################################
            # Here add your filters for the GDELT #
            # articles. For example, I want the   #
            # articles to be related to the US.   #
            #######################################

            scope_df = gd_df[
                ((gd_df['actor1countrycode'] == actor1countrycode) | (gd_df['actor2countrycode'] == actor2countrycode))]

            scope_df['n_keywords'] = scope_df['sourceurl_path'].str.findall('|'.join(self.keywords)).apply(len)

            scope_df = scope_df[scope_df['n_keywords'] > 0]
            scope_df = scope_df.drop_duplicates()

            scope_df['d_str'] = [d_str for _ in range(scope_df.shape[0])]

            if not n_articles: n_articles = scope_df.shape[0]

            #######################################################################################################
            # scope_df = scope_df[scope_df['sourceurl_path'].str.findall('|'.join(self.keywords)).apply(len) > 0] #
            #######################################################################################################

            scope_df = scope_df.sort_values(by=['n_keywords'], ascending=False)
            scope_df = scope_df.iloc[:n_articles]

            article_n = len(set(scope_df['sourceurl'].values))
            scope_df = list(scope_df.T.to_dict().values())

            sys.stdout.write('- Fetching {} articles for: {}'.format(article_n, d.strftime('%Y %m %d')))
            sys.stdout.flush()

            if article_n == 0: os.makedirs(os.path.join(self.output_dir, 'html/' + d_str), exist_ok=True)

            t0 = time.time()

            _ = list(enumerate(scope_df))

            pool = Pool(32)

            for i in tqdm(
                    pool.imap_unordered(self.article_collection_process, _),
                    desc='Fetching Article HTML',
                    total=len(_)
            ): pass

            pool.close()
            pool.join()

            t1 = time.time()

            sys.stdout.write(' [{}s]'.format(round(t1 - t0, 6)))
            sys.stdout.flush()
            print()

        file_paths = []

        for i in tqdm(list(range(self.duration))):
            d = self.from_date + timedelta(days=i)
            d_str = d.strftime('%Y%m%d')

            daily_path = os.path.join(self.output_dir, 'html/' + d_str + '/')

            file_paths += [os.path.join(daily_path, p) for p in os.listdir(daily_path)]

        pool = Pool(32)

        for i in tqdm(
                pool.imap_unordered(self.parse_html, file_paths),
                desc='HTML Parsing',
                total=len(file_paths)
        ): pass

        pool.close()
        pool.join()

    def parse_html(self, file_path):
        """
        Parse HTML content from a file.

        :param file_path: Path to the HTML file.
        :return: True if parsing is successful.
        """
        with open(file_path, 'r') as f:
            html_content = f.read()

        hgd = file_path.split('/')[-2]
        uid = file_path.split('/')[-1].replace('.html', '')

        if len(html_content) == 0: return None

        article = Article('', language='en')
        article.download(input_html=html_content)
        article.parse()

        hgd_dt = datetime.strptime(hgd, '%Y%m%d')

        article_dict = {
            'url': article.url,
            'uid': uid,
            'images': list(article.images),
            'publication-date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else hgd_dt.strftime(
                '%Y-%m-%d'),
            'text': article.text,
            'title': article.title,
            'top-image': article.top_image
        }

        article_dict_str = json.dumps(article_dict, indent=4)

        output_folder = os.path.join(self.output_dir, 'articles/' + hgd + '/')
        output_file = os.path.join(output_folder, uid + '.json')

        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)

        with open(output_file, 'w') as html_file:
            html_file.write(article_dict_str)

        return True


if __name__ == "__main__":
    keyword_extractor = URLKeywordExtractor(
        url_list=[
            'https://www.businessinsider.com/everything-you-need-to-know-about-chat-gpt-2023-1'
        ]
    )

    keywords = keyword_extractor.extract_keywords(n=20)
    keywords = ["openai", "gpt"]

    corpus_collector = NewsCorpusCollector(
        output_dir="./example",
        from_date=date(year=2023, month=8, day=15),
        to_date=date(year=2023, month=8, day=15),
        keywords=keywords
    )

    corpus_collector.collect_archives()
    corpus_collector.collect_articles()
    corpus_collector.pre_process_articles()
