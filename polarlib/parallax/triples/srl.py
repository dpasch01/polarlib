import os, itertools, multiprocessing, allennlp_models.tagging

from tqdm import tqdm
from multiprocessing import Pool
from polarlib.utils.utils import *
from nltk.tokenize import sent_tokenize, word_tokenize

from allennlp.predictors.predictor import Predictor

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class SRLTripleGenerator:

    def __init__(self, output_dir):

        self.output_dir = output_dir

        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

        self.article_paths   = list(itertools.chain.from_iterable([
            [os.path.join(o1, p) for p in o3]
            for o1, o2, o3 in os.walk(os.path.join(self.output_dir, 'pre_processed'))
        ]))

    def encode_sentence(self, sentence): return self.predictor.predict(sentence = sentence)

    def encode_sentences(self, sentences): return self.predictor.predict_batch_json([{'sentence': s} for s in sentences])

    def encode_article(self, path):

        try:

            article = load_article(path)

            output_folder = os.path.join(self.output_dir, 'parallax/srl/' + path.split('/')[-2])
            output_file   = os.path.join(output_folder, article['uid'] + '.json')

            if os.path.exists(output_file): return True

            sentence_list = article['text'].split('\n')
            sentence_list = [sent_tokenize(s) for s in sentence_list]
            sentence_list = list(itertools.chain.from_iterable(sentence_list))

            srl_triples = self.encode_sentences(sentence_list)

            srl_dict_str = json.dumps({
                'uid':     article['uid'],
                'triples': srl_triples.copy()
            })

            if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
            with open(output_file, 'w') as f:     json.dump(srl_dict_str, f)

        except Exception as ex:
            print(ex)
            return None

        return True

    def encode(self):

        pool = Pool(multiprocessing.cpu_count() - 8)

        for i in tqdm(
                pool.imap_unordered(self.encode_article, self.article_paths),
                desc  = 'Encode Triples with SRL',
                total = len(self.article_paths)
        ): pass

        pool.close()
        pool.join()

if __name__ == "__main__":

    srl_generator = SRLTripleGenerator('../example')

    sentences = ["Anthony Fauci gave a speech on emphasizing the need for a nationwide mask mandate."]

    for triples in srl_generator.encode_sentences(sentences):

        for t in triples['verbs']: print(t, '\n')