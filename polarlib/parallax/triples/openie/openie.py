import os, itertools, multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
from polarlib.utils.utils import *
from stanza.server import CoreNLPClient

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def encode_text(text):

    with CoreNLPClient(endpoint="http://localhost:9000", annotators=["openie"], be_quiet=True, start_server=False) as client:

        ann     = client.annotate(text)
        triples = []

        for sentence in ann.sentence:

            for triple in sentence.openieTriple:

                triples.append({
                    'subject':    triple.subject,
                    'relation':   triple.relation,
                    'object':     triple.object,
                    'confidence': triple.confidence
                })

        return triples

class OpenIETripleGenerator:

    def __init__(self, output_dir):

        self.output_dir = output_dir

        if not self.is_corenlp_installed():

            import stanza

            stanza.install_corenlp()

        self.client = CoreNLPClient(annotators=["openie"], be_quiet=False, threads=32)
        self.client.annotate("Is the CoreNLP server running?")

        self.article_paths   = list(itertools.chain.from_iterable([
            [os.path.join(o1, p) for p in o3]
            for o1, o2, o3 in os.walk(os.path.join(self.output_dir, 'pre_processed'))
        ]))

    def is_corenlp_installed(self):

        corenlp_home = os.environ.get('CORENLP_HOME', os.path.expanduser('~/stanza_corenlp'))

        return os.path.exists(corenlp_home)

    """
    def encode_text(self, text):

        with CoreNLPClient(endpoint=self.client.endpoint, annotators=["openie"], be_quiet=True, start_server=False) as client:

            ann     = client.annotate(text)
            triples = []

            for sentence in ann.sentence:

                for triple in sentence.openieTriple:

                    triples.append({
                        'subject':    triple.subject,
                        'relation':   triple.relation,
                        'object':     triple.object,
                        'confidence': triple.confidence
                    })

            return triples
    """

    def encode_article(self, path):

        try:

            article = load_article(path)

            output_folder = os.path.join(self.output_dir, 'parallax/openie/' + path.split('/')[-2])
            output_file   = os.path.join(output_folder, article['uid'] + '.json')

            if os.path.exists(output_file): return True

            article_text   = article['text']
            """openie_triples = self.encode_text(article_text)"""

            openie_triples = encode_text(article_text)

            openie_dict_str = json.dumps({
                'uid':     article['uid'],
                'triples': openie_triples.copy()
            })

            if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
            with open(output_file, 'w') as f:     json.dump(openie_dict_str, f)

        except Exception as ex:
            print(ex)
            return None

        return True

    def encode(self):

        pool = Pool(int(self.client.server.args[self.client.server.args.index('-threads') + 1]))

        for i in tqdm(
                pool.imap_unordered(self.encode_article, self.article_paths),
                desc  = 'Encode Triples with OpenIE',
                total = len(self.article_paths)
        ): pass

        pool.close()
        pool.join()

if __name__ == "__main__":

    openie_generator = OpenIETripleGenerator('../example')

    for t in openie_generator.encode_text("Anthony Fauci emphasizes the need for a nationwide mask mandate."):

        print('SUBJECT:     ', t['subject'])
        print('RELATIONSHIP:', t['relation'])
        print('OBJECT:      ', t['object'], '\n')
        print('CONFIDENCE:  ', t['confidence'],'\n')

