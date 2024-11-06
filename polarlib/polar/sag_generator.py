import os, pickle, multiprocessing, itertools, numpy, matplotlib.pyplot as plt, networkx as nx

from collections import Counter

from multiprocessing import Pool
from tqdm import tqdm

from collections import defaultdict

class SAGGenerator:

    def __init__(self, output_dir, entity_filter_list=None, entity_merge_dict={}):
        """
        Initialize the SAGGenerator.

        :param output_dir: The output directory where generated files will be stored.
        """
        self.attitude_path_list                   = []
        self.output_dir                           = output_dir
        self.pair_sentiment_attitude_dict         = {}
        self.encoded_pair_sentiment_attitude_dict = {}
        self.bins                                 = None

        self.entity_filter_list = entity_filter_list
        self.entity_merge_dict  = entity_merge_dict

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'attitudes')):

            for p in files: self.attitude_path_list.append(os.path.join(root, p))

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def _read_sentiment_attitudes(self, path):
        """
        Read sentiment attitudes from a given path.

        :param path: The path to the sentiment attitude file.
        :return: The entity attitudes from the sentiment attitude file.
        """
        with open(path, 'rb') as f: attidute_object = pickle.load(f)

        return attidute_object['attitudes']

    def process_sentiment_attitudes(self, file_list):
        pair_sentiment_attitude_dict = defaultdict(list)

        for attitude_path in file_list:
            result = self._read_sentiment_attitudes(attitude_path)

            for r in result:

                entity_attitudes = r['entity_attitudes']

                for p, attitudes in entity_attitudes.items():

                    p = tuple(sorted([
                        self.entity_merge_dict.get(p[0], p[0]),
                        self.entity_merge_dict.get(p[1], p[1])
                    ]))

                    if p[0] == p[1]: continue

                    if self.entity_filter_list:

                        if p[0] not in self.entity_filter_list or p[1] not in self.entity_filter_list: continue

                    if attitudes and not isinstance(attitudes[0], float):
                        pair_sentiment_attitude_dict[p].extend([
                            0 if t['NEUTRAL'] > max(t['NEGATIVE'], t['POSITIVE']) else
                            t['POSITIVE'] if t['POSITIVE'] > t['NEGATIVE'] else
                            -t['NEGATIVE'] for t in attitudes
                        ])
                    else:
                        pair_sentiment_attitude_dict[p].extend(attitudes)

        return pair_sentiment_attitude_dict

    def load_sentiment_attitudes(self):
        num_processes = min(multiprocessing.cpu_count(), len(self.attitude_path_list))
        pool          = multiprocessing.Pool(num_processes)

        chunk_size           = len(self.attitude_path_list) // num_processes
        attitude_path_chunks = [self.attitude_path_list[i:i + chunk_size] for i in range(0, len(self.attitude_path_list), chunk_size)]

        results = pool.map(self.process_sentiment_attitudes, attitude_path_chunks)
        pool.close()
        pool.join()

        pair_sentiment_attitude_dict = defaultdict(list)
        for result in results:
            for k, v in result.items():
                pair_sentiment_attitude_dict[k].extend(v)

        self.pair_sentiment_attitude_dict = dict(pair_sentiment_attitude_dict)

    def calculate_attitude_buckets(self, verbose=False, func=numpy.mean, filter_values=[]):
        """
        Calculate attitude buckets based on sentiment attitudes.

        :param verbose: If True, print additional statistics.
        :return: A list of attitude buckets.
        """
        attitude_population_list = list(itertools.chain.from_iterable([list(v) for v in self.pair_sentiment_attitude_dict.values()]))

        filtered_values = [a for a in attitude_population_list if a not in filter_values]

        if verbose:

            print('Mean:', numpy.mean(filtered_values))
            print('Std.:', numpy.std(filtered_values))
            print()

        filtered_lists = [list(filter(lambda a: a not in filter_values, v)) for v in self.pair_sentiment_attitude_dict.values()]
        non_empty_filtered_lists = [func(v) for v in filtered_lists if len(v) > 0]

        (n, bins, patches) = plt.hist(non_empty_filtered_lists, rwidth=0.95)

        self.bins = bins

        if verbose:

            for k, ij in enumerate(zip(self.bins[:-1], self.bins[1:])):

                i = ij[0]
                j = ij[1]

                print(f'{k}. >= {i:<25} and < {j:<25}:', n[k])

        self.bins = list(zip(bins[:-1], bins[1:]))

        return self.bins

    def convert_attitude_signs(self, bin_category_mapping, minimum_frequency=5, verbose=False, func=numpy.mean, filter_values=[]):
        """
        Convert sentiment attitudes to attitude signs based on bin category mapping.

        :param bin_category_mapping: A dictionary that maps bin categories to their respective bins.
        :param minimum_frequency: Minimum frequency threshold for pairs to be included.
        :param verbose: If True, print additional information.
        """

        ##############################################
        # Pre-process bin_category_mapping to create #
        # a direct mapping of bins to categories.    #
        ##############################################

        bin_to_category = {}
        for category, category_bins in bin_category_mapping.items():
            for b in category_bins: bin_to_category[b] = category

        def convert_sentiment_attitude(sentiment_value):

            for b, category in bin_to_category.items():

                if b[0] <= sentiment_value < b[1]: return category

            return "NEUTRAL"

        insufficient_pairs = {k for k, v in self.pair_sentiment_attitude_dict.items() if len(v) < minimum_frequency}

        self.encoded_pair_sentiment_attitude_dict = {
            k: convert_sentiment_attitude(func([a for a in v if a not in filter_values])) if len([a for a in v if a not in filter_values]) != 0 else 'NEUTRAL'
            for k, v in tqdm(self.pair_sentiment_attitude_dict.items())
            if k not in insufficient_pairs
        }

        if verbose:

            for l, f in Counter(self.encoded_pair_sentiment_attitude_dict.values()).most_common():

                print('{0:20} {1}'.format(l, f))

    def construct_sag(self):
        """
        Construct a Signed Attitude Graph (SAG) based on sentiment attitudes.

        :return: The constructed SAG, node mappings, and edge weights.
        """

        pair_frequency_dict                  = {k: len(v) for k, v in self.encoded_pair_sentiment_attitude_dict.items()}
        G, node_id, node_to_int, int_to_node = nx.Graph(), 0, {}, {}

        for p in tqdm(
                sorted(
                    self.encoded_pair_sentiment_attitude_dict.keys(),
                    key     = lambda k: pair_frequency_dict[k],
                    reverse = True
                )
        ):

            sentiment = self.encoded_pair_sentiment_attitude_dict[p]

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

    sag_generator = SAGGenerator('../example')

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