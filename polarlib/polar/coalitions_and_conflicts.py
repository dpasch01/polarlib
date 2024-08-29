import os, pickle, math, itertools, pandas as pd, numpy, subprocess, gzip

from tqdm import tqdm
from polarlib.utils.utils import *
from .frustration import *
from functools import partial

from multiprocessing import Pool
from collections import defaultdict, Counter
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import ward, fcluster

from collections import defaultdict

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class FellowshipExtractor:
    """
    A class for extracting fellowships from a signed network using SIMAP algorithm.

    Args:
        output_dir (str): The directory where output files will be saved.

    Attributes:
        output_dir (str): The directory where output files will be saved.
        fellowships (list): List to store extracted fellowships.
        sag (networkx.Graph): Signed adjacency graph.
        int_to_node (dict): Mapping of integer node indices to node names.
        node_to_int (dict): Mapping of node names to integer node indices.

    Raises:
        InsufficientSignedEdgesException: If there are not enough signed edges in the graph.

    Methods:
        _decode_fellowship_list(f_list, simap_iteration_dict):
            Decode fellowship list using the simap_iteration_dict.
        signed_network_clustering(resolution, verbose, jar_path):
            Run SIMAP algorithm to perform signed network clustering.
        extract_fellowships(n_iter, resolution, merge_iter, jar_path, verbose):
            Extract fellowships using iterations of signed network clustering.
        _extract_fellowships(n_iter, resolution, merge_iter, jar_path, verbose):
            Extract and merge fellowships using SIMAP algorithm.

    """
    def __init__(self, output_dir):

        self.output_dir = output_dir
        self.fellowships = []

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'rb') as f:         G = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'rb') as f: int_to_node = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'rb') as f: node_to_int = pickle.load(f)

        self.sag         = G
        self.int_to_node = int_to_node
        self.node_to_int = node_to_int

        negative_edges = []
        positive_edges = []

        for e in G.edges(data=True):

            if   e[2]['weight'] < 0: negative_edges.append(e)
            elif e[2]['weight'] > 0: positive_edges.append(e)

        if len(negative_edges) + len(positive_edges) == 0: raise InsufficientSignedEdgesException(self)

    def _decode_fellowship_list(self, f_list, simap_iteration_dict): return [simap_iteration_dict[int(index.split('_')[0])][int(index.split('_')[1])] for index in f_list]

    def signed_network_clustering(self, resolution=0.00, verbose=True, jar_path='./'):

        if os.path.isfile('/tmp/simap.wrapper.tsv'):

            if verbose: print('Removing simap previous data...', os.remove('/tmp/simap.wrapper.tsv'))
            else: os.remove('/tmp/simap.wrapper.tsv')

        if os.path.isfile('/tmp/simap.wrapper.partition.out'):

            if verbose: print('Removing simap previous partitions...', os.remove('/tmp/simap.wrapper.partition.out'))
            else: os.remove('/tmp/simap.wrapper.partition.out')

        _df_dict = [{'p_1': max(e[0], e[1]), 'p_2': min(e[1], e[0]), 'sign': int(e[2]['weight'])} for e in list(self.sag.edges(data=True))]

        _df = pd.DataFrame.from_dict(_df_dict)
        _df = _df.sort_values(by=['p_1'])

        if verbose: print('> Dumping graph in /tmp/simap.wrapper.tsv...')
        _df.to_csv('/tmp/simap.wrapper.tsv', sep='\t', index=False, header=False)

        subprocess_results = subprocess.run(
            ['java', '-jar', f'{jar_path}simap-1.0.0-final.jar', 'mdl', '-r', str(resolution), '-g',
             '/tmp/simap.wrapper.tsv', '-o', '/tmp/simap.wrapper.partition.out'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if verbose: print('> Errors: ', str(subprocess_results.stderr))
        if verbose: print('> Outputs: ', str(subprocess_results.stdout))

        _df_partitions = pd.read_csv('/tmp/simap.wrapper.partition.out', sep='\t', index_col=0, header=None)

        if verbose: print()

        return {k: v[1] for k, v in _df_partitions.T.to_dict().items()}

    def extract_fellowships(
            self,
            n_iter      = 25,
            resolution  = 0.05,
            merge_iter  = 10,
            jar_path    = './',
            output_flag = True,
            verbose     = False
    ):

        self.fellowships = self._extract_fellowships(
            n_iter     = n_iter,
            resolution = resolution,
            merge_iter = merge_iter,
            jar_path   = jar_path,
            verbose    = verbose
        )

        self.fellowships = sorted(self.fellowships, key=len, reverse=True)

        if verbose:

            for e in self.fellowships[0]: print('-', e)

        if output_flag:

            with open(os.path.join(self.output_dir, 'polarization/fellowships.json'), 'w') as f: json.dump({'fellowships': self.fellowships}, f)

        return self.fellowships

    def _extract_fellowships(
            self,
            n_iter     = 25,
            resolution = 0.05,
            merge_iter = 10,
            jar_path   = './',
            verbose    = False
    ):

        simap_iteration_dict = {i: [] for i in range(n_iter)}

        for iteration in tqdm(list(range(n_iter))):

            si_map_0 = self.signed_network_clustering(resolution=resolution, jar_path=jar_path, verbose=verbose)
            si_map_partitions = defaultdict(lambda: [])

            for k, v in si_map_0.items():             si_map_partitions[v].append(k)
            for k in list(si_map_partitions.keys()):  si_map_partitions[k] = list(si_map_partitions[k])

            si_map_partitions = dict(si_map_partitions)

            for i in range(len(si_map_partitions)):
                f_list = []

                for n in si_map_partitions[i]: f_list.append(self.int_to_node[n])

                simap_iteration_dict[iteration].append(f_list.copy())

        fellowship_indices = [['{}_{}'.format(i, j) for j, f in enumerate(f_list)] for i, f_list in
                              simap_iteration_dict.items()]
        fellowship_indices = list(itertools.chain.from_iterable(fellowship_indices))

        jaccard_indices = []

        for i, f1 in tqdm(list(enumerate(fellowship_indices)), desc='Extracting Fellowships using Monte Carlo'):
            x1 = int(f1.split('_')[0])
            y1 = int(f1.split('_')[1])

            j_f12 = []

            for j, f2 in enumerate(fellowship_indices):
                x2 = int(f2.split('_')[0])
                y2 = int(f2.split('_')[1])

                d12 = 1.0 - jaccard_index(simap_iteration_dict[x1][y1], simap_iteration_dict[x2][y2])

                j_f12.append(d12)

            jaccard_indices.append(j_f12)

        Z = ward(squareform(jaccard_indices))

        clusters = fcluster(Z, t=0.5, criterion='distance')

        cluster_dict = {}

        for i, c in enumerate(clusters):
            if c not in cluster_dict: cluster_dict[c] = []
            cluster_dict[c].append(fellowship_indices[i])

        max_freq_dict = {}

        for entry in sorted(cluster_dict.items(), key=lambda kv: len(kv[1]), reverse=True):

            f_list = self._decode_fellowship_list(entry[1], simap_iteration_dict)
            f_list = list(itertools.chain.from_iterable(f_list))

            for e, f in Counter(f_list).most_common():
                if e not in max_freq_dict:
                    max_freq_dict[e] = f
                else:
                    max_freq_dict[e] = max(max_freq_dict[e], f)

        visited, counter = [], 0
        merged_fellowships = []
        no_merge_fellowships = []

        for entry in sorted(cluster_dict.items(), key=lambda kv: len(kv[1]), reverse=True):

            f_list = self._decode_fellowship_list(entry[1], simap_iteration_dict)
            f_list = list(itertools.chain.from_iterable(f_list))

            merge = []
            no_merge = []

            for e, f in Counter(f_list).most_common():

                if e not in visited:
                    if f >= max_freq_dict[e]:
                        visited.append(e)
                        no_merge.append(e)
                        counter += 1

                if f >= merge_iter: merge.append(e)

            if len(merge) > 0:
                merged_fellowships.append(merge)
            else:
                no_merge_fellowships.append(no_merge)

        jaccard_indices = []

        for i, f1 in list(enumerate(merged_fellowships)):

            j_f12 = []

            for j, f2 in enumerate(merged_fellowships):
                d12 = 1.0 - jaccard_index(f1, f2)

                j_f12.append(d12)

            jaccard_indices.append(j_f12)

        clusters = []

        if len(jaccard_indices) > 0:
            Z = ward(squareform(jaccard_indices))
            clusters = fcluster(Z, t=0.25, criterion='distance')

        cluster_dict = {}

        for i, c in enumerate(clusters):
            if c not in cluster_dict: cluster_dict[c] = []
            cluster_dict[c].append(merged_fellowships[i])

        max_freq_dict = {}

        for entry in sorted(cluster_dict.items(), key=lambda kv: len(kv[1]), reverse=True):

            f_list = entry[1].copy()
            f_list = list(itertools.chain.from_iterable(f_list))

            for e, f in Counter(f_list).most_common():
                if e not in max_freq_dict: max_freq_dict[e] = f
                else: max_freq_dict[e] = max(max_freq_dict[e], f)

        re_merged_fellowships, visited = [], []

        for entry in sorted(cluster_dict.items(), key=lambda kv: len(list(itertools.chain.from_iterable(kv[1]))), reverse=True):

            f_list = list(itertools.chain.from_iterable(entry[1]))

            remerged = []
            for e, f in Counter(f_list).most_common():

                if e not in visited:
                    if f >= max_freq_dict[e]:
                        visited.append(e)
                        remerged.append(e)

            re_merged_fellowships.append(remerged)

        produced_fellowships = re_merged_fellowships + no_merge_fellowships
        produced_fellowships = [f for f in produced_fellowships if len(f) > 0]

        return produced_fellowships

class InsufficientSignedEdgesException(Exception):

    def __init__(self, extractor: FellowshipExtractor):
        super().__init__("Fellowship extraction failed due to insufficient signed edges in SAG: Number of nodes: {} | Number of edges: {}".format(
            extractor.sag.number_of_nodes(),
            extractor.sag.number_of_edges()
        ))

class DipoleGenerator:
    """
    A class for generating dipole information from fellowship graphs.

    Parameters:
        output_dir (str): The directory where the generated output will be stored.
    """
    def __init__(self, output_dir):
        """
        Initialize the DipoleGenerator.

        Args:
            output_dir (str): The output directory path.
        """
        self.output_dir = output_dir
        self.dipoles    = []

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'rb') as f:         G = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'rb') as f: int_to_node = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'rb') as f: node_to_int = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'fellowships.json'), 'r') as f:                    fellowship_list = json.load(f)['fellowships']

        self.sag = G
        self.int_to_node = int_to_node
        self.node_to_int = node_to_int
        self.fellowships = fellowship_list

    def get_connected_fellowships(self):
        """
        Get pairs of connected fellowships.

        Returns:
            set: Set of tuples representing pairs of connected fellowships.
        """
        community_membership = {}
        edge_set = set(self.sag.edges())
        connected_community_pairs = set()

        for community_idx, community in enumerate(self.fellowships):

            for node in community:
                node = self.node_to_int[node]
                community_membership[node] = community_idx

        for community in self.fellowships:

            for node in community:

                node = self.node_to_int[node]

                for neighbor in self.sag.neighbors(node):

                    if not neighbor in community_membership: continue

                    if community_membership[neighbor] != community_membership[node]:

                        if (node, neighbor) in edge_set or (neighbor, node) in edge_set:
                            connected_community_pairs.add(
                                tuple(sorted([community_membership[node], community_membership[neighbor]]))
                            )

        return connected_community_pairs

    def get_fellowship_graphs(self):
        """
        Get a list of fellowship graphs.

        Returns:
            list: List of NetworkX graphs representing fellowship graphs.
        """
        fellowship_graphs = []

        for f in self.fellowships:

            f_i = nx.Graph()

            for n in f: f_i.add_node(n, label=n)
            for e in self.sag.subgraph([self.node_to_int[n] for n in f]).edges(data=True):
                f_i.add_edge(self.int_to_node[e[0]], self.int_to_node[e[1]], weight=e[2]['weight'])

            fellowship_graphs.append(f_i.copy())

        return fellowship_graphs

    def extract_dipole(self, f_i_j, fellowship_graphs):
        """
        Extract dipole information from fellowship graphs.

        Args:
            f_i_j (tuple): Tuple of two fellowship indices.
            fellowship_graphs (list): List of fellowship graphs.

        Returns:
            None or tuple: None if dipole extraction fails, otherwise a tuple containing dipole information.
        """

        f_i, f_j = f_i_j
        int_nodes_1 = [self.node_to_int[n] for n in fellowship_graphs[f_i].nodes()]
        int_nodes_2 = [self.node_to_int[n] for n in fellowship_graphs[f_j].nodes()]
        d_ij = self.sag.subgraph(set(int_nodes_1 + int_nodes_2)).copy()

        positive_edges, negative_edges = [], []

        for e in d_ij.edges(data=True):
            if e[0] in int_nodes_1 and e[1] in int_nodes_1: continue
            if e[0] in int_nodes_2 and e[1] in int_nodes_2: continue

            if e[2]['weight'] > 0.0:
                positive_edges.append(e)
            elif e[2]['weight'] < 0.0:
                negative_edges.append(e)

        if (len(positive_edges) + len(negative_edges)) == 0: return None
        if len(negative_edges) == 0.0: return None

        p_positive = len(positive_edges) / (len(positive_edges) + len(negative_edges))
        p_negative = len(negative_edges) / (len(positive_edges) + len(negative_edges))

        dipole_g = nx.Graph()

        for n in d_ij.nodes(): dipole_g.add_node(self.int_to_node[n], label=self.int_to_node[n])

        for e in d_ij.edges(data=True): dipole_g.add_edge(self.int_to_node[e[0]], self.int_to_node[e[1]], weight=e[2]['weight'])

        return [(min(f_i, f_j), max(f_i, f_j)), {
            'd_ij':           dipole_g.copy(),
            'pos':            len(positive_edges),
            'neg':            len(negative_edges),
            'simap_1':        [self.int_to_node[n] for n in int_nodes_1],
            'simap_2':        [self.int_to_node[n] for n in int_nodes_2],
            'int_simap_1':    int_nodes_1,
            'int_simap_2':    int_nodes_2,
            'negative_ratio': p_negative,
            'positive_ratio': p_positive
        }]

    def calculate_frustration(self, d):
        """
        Calculate frustration index for a dipole.

        Args:
            d (tuple): Tuple containing dipole information.

        Returns:
            tuple: Tuple containing updated dipole information with frustration index.
        """

        si_f_g = None

        try:
            si_sign_G, si_adj_sign_G, si_sign_edgelist, si_int_to_node = G_to_fi(d[1]['d_ij'])
            si_f_g, si_f_e, si_t, si_solution_dict = calculate_frustration_index(si_sign_G, si_adj_sign_G, si_sign_edgelist)
        except Exception as ex:
            print(d[0], ex)

        d[1]['f_g'] = si_f_g

        return d

    def generate_dipoles(self, f_g_thr=0.7, n_r_thr=0.5):
        """
        Generate dipoles based on thresholds and save them to a file.

        Args:
            f_g_thr (float, optional): Frustration index threshold. Defaults to 0.7.
            n_r_thr (float, optional): Negative ratio threshold. Defaults to 0.5.
        """
        self.dipoles = self._generate_dipoles(f_g_thr=f_g_thr, n_r_thr=f_g_thr)
        with open(os.path.join(self.output_dir, 'polarization/' + 'dipoles.pckl'), 'wb') as f: pickle.dump(self.dipoles, f)

    def _generate_dipoles(self, f_g_thr=0.7, n_r_thr=0.5):
        """
        Generate dipoles based on thresholds.

        Args:
            f_g_thr (float, optional): Frustration index threshold. Defaults to 0.7.
            n_r_thr (float, optional): Negative ratio threshold. Defaults to 0.5.

        Returns:
            list: List of tuples containing dipole information.
        """
        f_i_j_list = self.get_connected_fellowships()

        fellowship_graphs = self.get_fellowship_graphs()

        pool = Pool(16)

        fellowship_dipoles = []

        _ = partial(
            self.extract_dipole,
            fellowship_graphs = fellowship_graphs
        )

        for result in tqdm(
                pool.imap_unordered(_, f_i_j_list),
                desc  = 'Extracting Dipoles',
                total = len(f_i_j_list)
        ):
            if result: fellowship_dipoles.append(result.copy())

        pool.close()
        pool.join()

        fellowship_dipoles = [d for d in fellowship_dipoles if d]
        fellowship_dipoles = [d for d in fellowship_dipoles if d and d[1]['negative_ratio'] >= n_r_thr]
        fellowship_dipoles = [self.calculate_frustration(d) for d in tqdm(fellowship_dipoles, desc='Generating Dipoles')]
        fellowship_dipoles = [d for d in fellowship_dipoles if d[1]['f_g'] == None or d[1]['f_g'] >= f_g_thr]

        return fellowship_dipoles

class TopicAttitudeCalculator:
    """
    A class for calculating topic attitudes and polarization indices based on sentiment attitudes and dipole information.
    """

    def __init__(self, output_dir, entity_filter_list=[], entity_merge_dict={}):
        """
        Initializes the TopicAttitudeCalculator with the specified output directory.

        :param output_dir: The directory where output data will be saved.
        """
        self.output_dir = output_dir

        self.entity_filter_list = entity_filter_list
        self.entity_merge_dict  = entity_merge_dict

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'rb') as f:         G = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'rb') as f: int_to_node = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'rb') as f: node_to_int = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'fellowships.json'), 'r') as f:  fellowship_list = json.load(f)['fellowships']
        with gzip.open(os.path.join(self.output_dir, 'topics.json.gz'), 'r') as f:            topics = json.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'dipoles.pckl'), 'rb') as f:     dipole_list = pickle.load(f)

        self.sag = G
        self.int_to_node = int_to_node
        self.node_to_int = node_to_int
        self.fellowships = fellowship_list
        self.topics      = topics
        self.dipoles     = dipole_list

        self.dipole_topics_dict            = {}
        self.entity_np_sentiment_attitudes = {}

        self.clean_np_dict      = {}
        self.np_topics_dict     = {}
        self.attitude_path_list = []

        self.dipole_topic_attitudes = []

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'attitudes')):

            for p in files: self.attitude_path_list.append(os.path.join(root, p))

        for t in self.topics:

            for np, c in zip(self.topics[t]['noun_phrases'], self.topics[t]['pre_processed']):

                    self.clean_np_dict[np] = c

                    if c not in self.np_topics_dict: self.np_topics_dict[c] = []

                    self.np_topics_dict[c].append(t)

                    self.np_topics_dict[c] = list(set(self.np_topics_dict[c]))

    def undersample_dipole_attitudes(self, dipole_tuple, aggr_func=numpy.mean, verbose=False):
        """
        Undersamples the dipole attitudes by aggregating sentiment attitudes.

        :param dipole_tuple: A tuple containing the dipole information and sentiment attitudes.
        :param aggr_func: Aggregation function for sentiment attitudes.
        :return: A list of polarized clusters and their attitude scores.
        """
        fi, fj = dipole_tuple[0]
        dipole_dict = dipole_tuple[1]
        dipole_key = (fi, fj)

        print(dipole_key)

        if dipole_key not in self.dipole_topics_dict: return []

        print(dipole_key)

        fi_entities = dipole_dict['simap_1']
        fj_entities = dipole_dict['simap_2']

        entity_np_sentiment_attitudes = self.entity_np_sentiment_attitudes

        def aggregate_attitudes(entities):
            attitudes_dict = defaultdict(list)
            for e in entities:
                for np, atts in entity_np_sentiment_attitudes.get(e, {}).items():
                    attitudes_dict[np].extend(atts)
            return attitudes_dict

        if verbose: print('Calculating attitude aggregates...')

        fi_np_attitudes_dict = aggregate_attitudes(fi_entities)
        fj_np_attitudes_dict = aggregate_attitudes(fj_entities)

        dipole_topics = self.dipole_topics_dict[dipole_key]
        np_clusters = dipole_topics['np_clusters']
        np_to_ci    = {np: ci for ci, nps in np_clusters.items() for np in nps}

        if verbose: print('Done.')

        def frame_attitudes(np_attitudes_dict):
            frame_attitudes_dict = defaultdict(list)
            for np, np_atts in np_attitudes_dict.items():
                ci = np_to_ci.get(np)
                if ci:  frame_attitudes_dict[ci].extend(np_atts)
            return frame_attitudes_dict

        if verbose: print('Framing attitudes...')

        fi_frame_attitudes = frame_attitudes(fi_np_attitudes_dict)
        fj_frame_attitudes = frame_attitudes(fj_np_attitudes_dict)

        if verbose: print('Done.')

        polarization_list = [
            {
                'dipole': dipole_key,
                'atts_fi': fi_frame_attitudes[ci],
                'atts_fj': fj_frame_attitudes[ci],
                'topic': {
                    'id': ci,
                    'nps': np_clusters[ci]
                }
            }
            for ci in np_clusters if ci in fi_frame_attitudes and ci in fj_frame_attitudes
        ]

        return polarization_list

    def resample_attitudes(self, atts, n):
        """
        Resamples attitudes to achieve a balanced distribution.

        :param atts: A list of attitudes.
        :param n: The desired number of resampled attitudes.
        :return: A list of resampled attitudes.
        """
        total_v, v_ratios = len(atts), {}

        for v in Counter(atts).most_common(): v_ratios[v[0]] = v[1] / total_v
        r_atts = list(
            itertools.chain.from_iterable([[v for i in range(math.floor(n * v_ratios[v]))] for v in v_ratios]))

        return r_atts

    def load_sentiment_attitudes(self):
        """
        Loads sentiment attitudes from stored attitude files and associates them with noun phrases.

        This method uses multiprocessing for efficient loading of attitude data.

        :return: None
        """
        num_processes = multiprocessing.cpu_count() // 2  # Adjust based on testing
        chunk_size = len(self.attitude_path_list) // num_processes

        pool = Pool(num_processes)

        entity_np_sentiment_attitudes = defaultdict(lambda: defaultdict(list))

        for result in tqdm(
                pool.imap_unordered(self.read_sentiment_attitudes, self.attitude_path_list, chunksize=chunk_size),
                desc='Fetching Attitudes',
                total=len(self.attitude_path_list)
        ):
            for r in result:
                noun_phrase_attitudes = r['noun_phrase_attitudes']

                for (entity, np), attitudes in noun_phrase_attitudes.items():

                    entity = self.entity_merge_dict.get(entity, entity)

                    if self.entity_filter_list and entity not in self.entity_filter_list: continue

                    clean_np = self.clean_np_dict.get(np)

                    if not clean_np: continue

                    if attitudes and not isinstance(attitudes[0], float):
                        sentiment_values = [
                            0 if t['NEUTRAL'] > max(t['NEGATIVE'], t['POSITIVE']) else
                            t['POSITIVE'] if t['POSITIVE'] > t['NEGATIVE'] else
                            -t['NEGATIVE'] for t in attitudes
                        ]
                        entity_np_sentiment_attitudes[entity][clean_np].extend(sentiment_values)
                    else:
                        entity_np_sentiment_attitudes[entity][clean_np].extend(attitudes)

        self.entity_np_sentiment_attitudes = dict(entity_np_sentiment_attitudes)

        pool.close()
        pool.join()

    def read_sentiment_attitudes(self, path):
        """
        Reads sentiment attitude data from a file.

        :param path: The path to the sentiment attitude file.
        :return: A dictionary containing sentiment attitudes for noun phrases.
        """
        with open(path, 'rb') as f: attidute_object = pickle.load(f)

        return attidute_object['attitudes']

    def extract_dipole_topics(self, dipole_tuple):
        """
        Extracts dipole-related topic information, including attitudes and clusters.

        :param dipole_tuple: A tuple containing dipole information and sentiment attitudes.
        :return: A dictionary containing dipole-related topic information.
        """
        dipole_id, dipole_obj = dipole_tuple[0], dipole_tuple[1]
        np_attitudes_dict = {}

        entity_np_sentiment_attitudes = self.entity_np_sentiment_attitudes
        clean_np_dict = self.clean_np_dict
        np_topics_dict = self.np_topics_dict

        for entity in dipole_obj['d_ij'].nodes():
            entity_attitudes = entity_np_sentiment_attitudes.get(entity)
            if not entity_attitudes: 
                print(f'No attitudes found for {entity}')
                continue

            for np, att_obj in entity_attitudes.items():
                np_attitudes_dict.setdefault(np, []).extend(att_obj)

        dipole_np_labels = {
            np: set(np_topics_dict[clean_np_dict[np]])
            for np in np_attitudes_dict.keys()
            if np in clean_np_dict and clean_np_dict[np] in np_topics_dict
        }

        if not dipole_np_labels:
            print('Not dipole NP labels found.\n')
            return None

        cluster_dict = defaultdict(list)
        for np, topics in dipole_np_labels.items():
            for topic in topics:
                cluster_dict[topic].append(np)

        return {
            'fellowship_1': dipole_id[0],
            'fellowship_2': dipole_id[1],
            'dipole_topics': {
                'np_attitudes': np_attitudes_dict,
                'np_clusters': dict(cluster_dict)
            }
        }

    def get_polarization_topics(self):
        """
        Retrieves dipole-related topics and organizes them in a dictionary.

        :return: A dictionary containing dipole-related topic information.
        """
        dipole_topics = []

        for dipole in tqdm(self.dipoles):

            if not dipole: continue

            d_topics = self.extract_dipole_topics(dipole)

            if not d_topics: 
                 print('No topics found in dipole.')
                 continue

            dipole_topics.append(d_topics)

        self.dipole_topics_dict = {
            (d['fellowship_1'], d['fellowship_2']): d['dipole_topics']
            for d in dipole_topics if d
        }

        return self.dipole_topics_dict

    def calculate_polarization_index(self, atts):

        A_minus = [t for t in atts if t < 0.0]
        A_plus = [t for t in atts if t > 0.0]

        if (len(A_minus) + len(A_plus)) == 0.0: return 0.0

        D_A = abs(
            (len(A_plus) / (len(A_plus) + len(A_minus))) - \
            (len(A_minus) / (len(A_plus) + len(A_minus)))
        )

        gc_minus = numpy.mean(A_minus) if len(A_minus) > 0 else 0.0
        gc_plus = numpy.mean(A_plus) if len(A_plus) > 0 else 0.0

        gc_d = (abs(gc_plus - gc_minus)) / 2

        m = (1.0 - D_A) * gc_d

        return m

    def get_topic_attitudes(self, aggr_func=numpy.mean):
        """
        Retrieves topic attitudes and polarization indices for dipoles.

        :param aggr_func: Aggregation function for sentiment attitudes.
        :return: A list of dictionaries containing topic attitudes and polarization indices for dipoles.
        """

        dipole_topic_attitudes = []

        for dipole in tqdm(self.dipoles, desc='Undersampling Topic Attitudes'):

            if not dipole: continue

            dipole_topic_attitudes.append(self.undersample_dipole_attitudes(dipole, aggr_func))

        dipole_topic_attitudes = list(itertools.chain.from_iterable(dipole_topic_attitudes))

        # print(dipole_topic_attitudes[0])

        filtered_topic_attitudes = []

        for dipole_t in dipole_topic_attitudes:
            if len(set(dipole_t['atts_fi'])) == 1 and dipole_t['atts_fi'][0] == 0.0: continue
            if len(set(dipole_t['atts_fj'])) == 1 and dipole_t['atts_fj'][0] == 0.0: continue

            filtered_topic_attitudes.append(dipole_t.copy())

        for i, dipole_t in tqdm(list(enumerate(filtered_topic_attitudes)), desc='Extracting Topical Attitudes'):

            ################################################################
            # Remove any 0.0 attitudes from Fi and Fj for the resampling.  #
            # This code might also remove from original dipole_t object.   #
            ################################################################

            dipole_t['atts_fi'] = [v for v in dipole_t['atts_fi'] if v != 0.0]
            dipole_t['atts_fj'] = [v for v in dipole_t['atts_fj'] if v != 0.0]

            if len(dipole_t['atts_fi']) == 0 or len(dipole_t['atts_fj']) == 0: continue

            ###########################################################
            # If Fi and Fj attitudes have the same size then they do  #
            # not need resampling.                                    #
            ###########################################################

            if len(dipole_t['atts_fi']) == len(dipole_t['atts_fj']):
                filtered_topic_attitudes[i]['X']  = dipole_t['atts_fi'] + dipole_t['atts_fj']
                filtered_topic_attitudes[i]['pi'] = self.calculate_polarization_index(filtered_topic_attitudes[i]['X'])
            else:

                if len(dipole_t['atts_fi']) > len(dipole_t['atts_fj']):

                    fj_res = self.resample_attitudes(dipole_t['atts_fj'], len(dipole_t['atts_fi']))

                    filtered_topic_attitudes[i]['X']      = dipole_t['atts_fi'] + dipole_t['atts_fj']
                    filtered_topic_attitudes[i]['X_res']  = dipole_t['atts_fi'] + fj_res
                    filtered_topic_attitudes[i]['pi']     = self.calculate_polarization_index(filtered_topic_attitudes[i]['X'])
                    filtered_topic_attitudes[i]['pi_res'] = self.calculate_polarization_index(filtered_topic_attitudes[i]['X_res'])

                else:

                    fi_res = self.resample_attitudes(dipole_t['atts_fi'], len(dipole_t['atts_fj']))

                    filtered_topic_attitudes[i]['X']      = dipole_t['atts_fi'] + dipole_t['atts_fj']
                    filtered_topic_attitudes[i]['X_res']  = dipole_t['atts_fj'] + fi_res
                    filtered_topic_attitudes[i]['pi']     = self.calculate_polarization_index(filtered_topic_attitudes[i]['X'])
                    filtered_topic_attitudes[i]['pi_res'] = self.calculate_polarization_index(filtered_topic_attitudes[i]['X_res'])

        self.dipole_topic_attitudes = filtered_topic_attitudes

        with open(os.path.join(self.output_dir, 'polarization/attitudes.pckl'), 'wb') as f: pickle.dump(self.dipole_topic_attitudes, f)

        return filtered_topic_attitudes

if __name__ == "__main__":

    fellowship_extractor = FellowshipExtractor('../example')

    fellowships          = fellowship_extractor.extract_fellowships(
        n_iter     = 1,
        resolution = 0.05,
        merge_iter = 1,
        jar_path   ='../../',
        verbose    = True
    )

    dipole_generator = DipoleGenerator('../example')
    dipoles          = dipole_generator.generate_dipoles(f_g_thr=0.7, n_r_thr=0.5)

    topic_attitude_calculator = TopicAttitudeCalculator('../example')
    topic_attitude_calculator.load_sentiment_attitudes()

    topic_attitudes           = topic_attitude_calculator.get_topic_attitudes()
