import os, pickle, multiprocessing, itertools, numpy, matplotlib.pyplot as plt, networkx as nx

from collections import Counter

from multiprocessing import Pool
from tqdm import tqdm


class SAGGenerator:

    def __init__(self, output_dir):

        self.attitude_path_list           = []
        self.output_dir                   = output_dir
        self.pair_sentiment_attitude_dict = {}
        self.bins                         = None

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'attitudes')):

            for p in files: self.attitude_path_list.append(os.path.join(root, p))

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _read_sentiment_attitudes(self, path):

        with open(path, 'rb') as f: attidute_object = pickle.load(f)

        return attidute_object['entity_attitudes']

    def load_sentiment_attitudes(self):

        pool = Pool(multiprocessing.cpu_count() - 4)

        for result in tqdm(
                pool.imap_unordered(self._read_sentiment_attitudes, self.attitude_path_list),
                desc  = 'Fetching Attitudes',
                total = len(self.attitude_path_list)
        ):
            for p in result:

                if p not in self.pair_sentiment_attitude_dict: self.pair_sentiment_attitude_dict[p] = []
                self.pair_sentiment_attitude_dict[p] += [
                    0 if t['NEUTRAL'] > t['NEGATIVE'] and t['NEUTRAL'] > t['POSITIVE'] else \
                         t['POSITIVE'] if t['POSITIVE'] > t['NEGATIVE'] and t['POSITIVE'] > t['NEUTRAL'] else \
                        -t['NEGATIVE'] for t in result[p]
                ]

        pool.close()
        pool.join()

    def calculate_attitude_buckets(self, verbose=False):

        attitude_population_list = list(itertools.chain.from_iterable([list(v) for v in self.pair_sentiment_attitude_dict.values()]))

        if verbose:

            print('Mean:', numpy.mean(attitude_population_list))
            print('Std.:', numpy.std(attitude_population_list))
            print()

        (n, bins, patches) = plt.hist([numpy.mean(list(v)) for v in self.pair_sentiment_attitude_dict.values()], rwidth=0.95)

        self.bins = bins

        if verbose:

            for k, ij in enumerate(zip(self.bins[:-1], self.bins[1:])):

                i = ij[0]
                j = ij[1]

                print(f'>= {i:<25} and < {j:<25}:', n[k])

        self.bins = list(zip(bins[:-1], bins[1:]))

        return self.bins

    def convert_attitude_signs(self, bin_category_mapping, minimum_frequency=5, verbose=False):

        """
        bin_category_mapping = {
            "NEGATIVE":  [bins[0], bins[1], bins[2], bins[3]],
            "NEUTRAL": [bins[4], bins[5]],
            "POSITIVE": [bins[6], bins[7], bins[8], bins[9]]
        }
        :return:
        """

        def convert_sentiment_attitude(sentiment_value):

            sentiment_category = "NEUTRAL"

            for category, category_bins in bin_category_mapping.items():

                for b in category_bins:

                    if sentiment_value >= b[0] and sentiment_value < b[1]:
                        sentiment_category = category
                        break

            return sentiment_category

        pair_frequency_dict = {k: len(v) for k, v in self.pair_sentiment_attitude_dict.items()}
        freq                = numpy.percentile(sorted(pair_frequency_dict.values()), 75)

        insufficient_pair_list = []

        for p, v in pair_frequency_dict.items():

            if v < minimum_frequency: insufficient_pair_list.append(p)

        self.pair_sentiment_attitude_dict = {
            k: convert_sentiment_attitude(numpy.mean(v))
            for k,v in tqdm(self.pair_sentiment_attitude_dict.items())
            if k not in insufficient_pair_list
        }

        if verbose:

            for l, f in Counter(list(self.pair_sentiment_attitude_dict.values())).most_common():

                print('{0:20} {1}'.format(l, f))

    def construct_sag(self):

        pair_frequency_dict                  = {k: len(v) for k, v in self.pair_sentiment_attitude_dict.items()}
        G, node_id, node_to_int, int_to_node = nx.Graph(), 0, {}, {}

        for p in tqdm(
                sorted(
                    self.pair_sentiment_attitude_dict.keys(),
                    key     = lambda k: pair_frequency_dict[k],
                    reverse = True
                )
        ):

            sentiment = self.pair_sentiment_attitude_dict[p]

            if sentiment == "NEUTRAL": continue

            if not p[0] in node_to_int:
                node_to_int[p[0]]    = node_id
                int_to_node[node_id] = p[0]
                node_id             += 1

            if not p[1] in node_to_int:
                node_to_int[p[1]]    = node_id
                int_to_node[node_id] = p[1]
                node_id             += 1

            p_1, p_2 = node_to_int[p[0]], node_to_int[p[1]]

            G.add_edge(p_1, p_2, weight=-1 if sentiment == "NEGATIVE" else 1)

        index, node_freq_dict = 1, {}

        for n in tqdm(G.nodes()):
            n1 = int_to_node[n]

            f1 = 0.0

            for n2 in G.neighbors(n):
                n2 = int_to_node[n2]

                p = [n1, n2]
                p.sort()
                p = (p[0], p[1])

                f12 = pair_frequency_dict[p]
                f1 += f12

            node_freq_dict[n1] = f1

        import os, pickle

        if os.path.exists(os.path.join(self.output_dir, 'polarization/')): print('File already exists.')
        else: os.makedirs(os.path.join(self.output_dir, 'polarization/'))

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'wb') as f:         pickle.dump(G, f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'wb') as f: pickle.dump(int_to_node, f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'wb') as f: pickle.dump(node_to_int, f)

        return G, node_to_int, int_to_node

if __name__ == "__main__":

    sag_generator = SAGGenerator('./example')

    sag_generator.load_sentiment_attitudes()

    bins = sag_generator.calculate_attitude_buckets(verbose=True)

    sag_generator.convert_attitude_signs(
        bin_category_mapping = {
            "NEGATIVE":  [bins[0], bins[1], bins[2], bins[3]],
            "NEUTRAL": [bins[4], bins[5]],
            "POSITIVE": [bins[6], bins[7], bins[8], bins[9]]
        },
        verbose              = True
    )

    sag_generator.construct_sag()