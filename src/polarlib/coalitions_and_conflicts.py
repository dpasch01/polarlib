import os, pickle, networkx as nx, math, itertools, pandas as pd, json, scipy.stats, jsonpickle, numpy, networkx as nx, subprocess, multiprocessing

from tqdm import tqdm
from utilities import *
from frustration import *
from functools import partial

from multiprocessing import Pool
from ast import literal_eval as make_tuple
from collections import defaultdict, Counter
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import ward, fcluster

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class FellowshipExtractor:

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
            n_iter     = 25,
            resolution = 0.05,
            merge_iter = 10,
            jar_path   = './',
            verbose    = False
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

        with open(os.path.join(self.output_dir, 'fellowships.json'), 'w') as f: json.dump({'fellowships': self.fellowships}, f)

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
            for k in list(si_map_partitions.keys()): si_map_partitions[k] = list(si_map_partitions[k])

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

    def __init__(self, output_dir):

        self.output_dir = output_dir
        self.dipoles    = []

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'rb') as f:         G = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'rb') as f: int_to_node = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'rb') as f: node_to_int = pickle.load(f)
        with open(os.path.join(self.output_dir, 'fellowships.json'), 'r') as f:                    fellowship_list = json.load(f)['fellowships']

        self.sag = G
        self.int_to_node = int_to_node
        self.node_to_int = node_to_int
        self.fellowships = fellowship_list

    def get_connected_fellowships(self):

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

                    if community_membership[neighbor] != community_membership[node]:

                        if (node, neighbor) in edge_set or (neighbor, node) in edge_set:
                            connected_community_pairs.add(
                                tuple(sorted([community_membership[node], community_membership[neighbor]]))
                            )

        return connected_community_pairs

    def get_fellowship_graphs(self):

        fellowship_graphs = []

        for f in self.fellowships:

            f_i = nx.Graph()

            for n in f: f_i.add_node(n, label=n)
            for e in self.sag.subgraph([self.node_to_int[n] for n in f]).edges(data=True):
                f_i.add_edge(self.int_to_node[e[0]], self.int_to_node[e[1]], weight=e[2]['weight'])

            fellowship_graphs.append(f_i.copy())

        return fellowship_graphs

    def extract_dipole(self, f_i_j, fellowship_graphs):

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

        si_sign_G, si_adj_sign_G, si_sign_edgelist, si_int_to_node = G_to_fi(d[1]['d_ij'])
        si_f_g, si_f_e, si_t, si_solution_dict = calculate_frustration_index(si_sign_G, si_adj_sign_G, si_sign_edgelist)

        d[1]['f_g'] = si_f_g

        return d

    def generate_dipoles(self, f_g_thr=0.7, n_r_thr=0.5):

        self.dipoles = self._generate_dipoles(f_g_thr=f_g_thr, n_r_thr=f_g_thr)
        with open(os.path.join(self.output_dir, 'dipoles.pckl'), 'wb') as f: pickle.dump(self.dipoles, f)

    def _generate_dipoles(self, f_g_thr=0.7, n_r_thr=0.5):

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
        fellowship_dipoles = [d for d in fellowship_dipoles if d[1]['f_g'] >= f_g_thr]

        return fellowship_dipoles

class TopicAttitudeCalculator:

    def __init__(self, output_dir):

        self.output_dir = output_dir

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'rb') as f:         G = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'rb') as f: int_to_node = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'rb') as f: node_to_int = pickle.load(f)
        with open(os.path.join(self.output_dir, 'fellowships.json'), 'r') as f:                    fellowship_list = json.load(f)['fellowships']
        with open(os.path.join(self.output_dir, 'topics.json'), 'r') as f:                         topics = json.load(f)
        with open(os.path.join(self.output_dir, 'dipoles.pckl'), 'rb') as f:                       dipole_list = pickle.load(f)

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

    def undersample_dipole_attitudes(
            self,
            dipole_tuple,
            aggr_func=numpy.mean
    ):
        fi, fj      = dipole_tuple[0]
        dipole_dict = dipole_tuple[1]

        if (fi, fj) not in self.dipole_topics_dict: return []

        fi_entities = dipole_dict['simap_1']
        fj_entities = dipole_dict['simap_2']

        fi_np_attitudes_dict = {}
        fj_np_attitudes_dict = {}

        for e in fi_entities:
            if e not in self.entity_np_sentiment_attitudes: continue

            for np, atts in self.entity_np_sentiment_attitudes[e].items():
                if not np in fi_np_attitudes_dict: fi_np_attitudes_dict[np] = []

                fi_np_attitudes_dict[np] += atts

        for e in fj_entities:
            if e not in self.entity_np_sentiment_attitudes: continue

            for np, atts in self.entity_np_sentiment_attitudes[e].items():
                if not np in fj_np_attitudes_dict: fj_np_attitudes_dict[np] = []

                fj_np_attitudes_dict[np] += atts

        fi_frame_attitudes = {}
        fj_frame_attitudes = {}

        for np, np_atts in fi_np_attitudes_dict.items():

            for ci in self.dipole_topics_dict[(fi, fj)]['np_clusters']:
                if np not in self.dipole_topics_dict[(fi, fj)]['np_clusters'][ci]: continue
                if ci not in fi_frame_attitudes: fi_frame_attitudes[ci] = []
                fi_frame_attitudes[ci] += np_atts

        for np, np_atts in fj_np_attitudes_dict.items():

            for ci in self.dipole_topics_dict[(fi, fj)]['np_clusters']:
                if np not in self.dipole_topics_dict[(fi, fj)]['np_clusters'][ci]: continue
                if ci not in fj_frame_attitudes: fj_frame_attitudes[ci] = []
                fj_frame_attitudes[ci] += np_atts

        polarization_list = []

        for ci in self.dipole_topics_dict[(fi, fj)]['np_clusters']:
            if ci not in fi_frame_attitudes: continue
            if ci not in fj_frame_attitudes: continue

            polarization_list.append({
                'dipole':  (fi, fj),
                'atts_fi': fi_frame_attitudes[ci],
                'atts_fj': fj_frame_attitudes[ci],
                'topic': {
                    'id':  ci,
                    'nps': self.dipole_topics_dict[(fi, fj)]['np_clusters'][ci]
                }
            })

        return polarization_list

    def resample_attitudes(self, atts, n):
        total_v, v_ratios = len(atts), {}

        for v in Counter(atts).most_common(): v_ratios[v[0]] = v[1] / total_v
        r_atts = list(
            itertools.chain.from_iterable([[v for i in range(math.floor(n * v_ratios[v]))] for v in v_ratios]))

        return r_atts

    def load_sentiment_attitudes(self):

        pool = Pool(multiprocessing.cpu_count() - 4)

        for result in tqdm(
                pool.imap_unordered(self.read_sentiment_attitudes, self.attitude_path_list),
                desc='Fetching Attitudes',
                total=len(self.attitude_path_list)
        ):
            for p in result:

                if p[1] not in self.clean_np_dict: break

                if p[0] not in self.entity_np_sentiment_attitudes: self.entity_np_sentiment_attitudes[p[0]] = {}

                c = self.clean_np_dict[p[1]]

                if c not in self.entity_np_sentiment_attitudes[p[0]]: self.entity_np_sentiment_attitudes[p[0]][c] = []

                self.entity_np_sentiment_attitudes[p[0]][c] += [
                    0 if t['NEUTRAL'] > t['NEGATIVE'] and t['NEUTRAL'] > t['POSITIVE'] else \
                         t['POSITIVE'] if t['POSITIVE'] > t['NEGATIVE'] and t['POSITIVE'] > t['NEUTRAL'] else \
                        -t['NEGATIVE'] for t in result[p]
                ]

        pool.close()
        pool.join()

    def read_sentiment_attitudes(self, path):

        with open(path, 'rb') as f: attidute_object = pickle.load(f)

        return attidute_object['noun_phrase_attitudes']

    def extract_dipole_topics(self, dipole_tuple):

        dipole_id, dipole_obj, np_attitudes_dict = dipole_tuple[0], dipole_tuple[1], {}

        for entity in dipole_obj['d_ij'].nodes():

            if entity not in self.entity_np_sentiment_attitudes: continue

            for np, att_obj in self.entity_np_sentiment_attitudes[entity].items():

                if np not in np_attitudes_dict: np_attitudes_dict[np] = []

                np_attitudes_dict[np] += att_obj.copy()

        dipole_np_list = list(sorted(np_attitudes_dict.keys()))
        dipole_np_labels = {np: set(self.np_topics_dict[self.clean_np_dict[np]]) for np in dipole_np_list if np in self.clean_np_dict and self.clean_np_dict[np] in self.np_topics_dict}

        np_attitudes_dict = {k: v for k, v in np_attitudes_dict.items() if k in dipole_np_labels}.copy()

        if len(dipole_np_labels) == 0: return None

        cluster_dict = {}

        for k, v in dipole_np_labels.items():

            for _ in v:
                if _ not in cluster_dict: cluster_dict[_] = []
                cluster_dict[_].append(k)

        return {
            'fellowship_1': dipole_id[0],
            'fellowship_2': dipole_id[1],
            'dipole_topics': {
                'np_attitudes': np_attitudes_dict.copy(),
                'np_clusters': dict(cluster_dict).copy()
            }
        }

    def get_polarization_topics(self):

        dipole_topics = []

        for dipole in self.dipoles:

            if not dipole: continue

            d_topics = self.extract_dipole_topics(dipole)

            if not d_topics: continue

            dipole_topics.append(d_topics)

        self.dipole_topics_dict = {
            (d['fellowship_1'], d['fellowship_2']): d['dipole_topics']
            for d in dipole_topics if d
        }

        return self.dipole_topics_dict

    def get_topic_attitudes(self, aggr_func=numpy.mean):

        self.get_polarization_topics()

        dipole_topic_attitudes = []

        for dipole in tqdm(self.dipoles, desc='Undersampling Topic Attitudes'):
            if not dipole: continue

            dipole_topic_attitudes.append(self.undersample_dipole_attitudes(dipole, aggr_func))

        dipole_topic_attitudes = list(itertools.chain.from_iterable(dipole_topic_attitudes))

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

        with open(os.path.join(self.output_dir, 'attitudes.pckl'), 'wb') as f: pickle.dump(self.dipole_topic_attitudes, f)

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