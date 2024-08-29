import os, json, pickle, networkx as nx, numpy, multiprocessing, itertools, gzip

from tqdm import tqdm
from multiprocessing import Pool

from polarlib.utils.utils import  *

class PolarizationKnowledgeGraph:

    def __init__(self, output_dir):

        self.output_dir = output_dir
        self.pkg        = None
        self.polarization_mapping = None

        with open(os.path.join(self.output_dir, 'polarization/' + 'sag.pckl'), 'rb') as f:         self.sag             = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'int_to_node.pckl'), 'rb') as f: self.int_to_node     = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'node_to_int.pckl'), 'rb') as f: self.node_to_int     = pickle.load(f)

        with open(os.path.join(self.output_dir, 'polarization/' + 'fellowships.json'), 'r') as f:                    self.fellowship_list = json.load(f)['fellowships']
        with gzip.open(os.path.join(self.output_dir, 'topics.json.gz'), 'r') as f:                              self.topic_dict      = json.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'dipoles.pckl'), 'rb') as f:                       self.dipole_list     = pickle.load(f)
        with open(os.path.join(self.output_dir, 'polarization/' + 'attitudes.pckl'), 'rb') as f:                     self.attitude_list   = pickle.load(f)

    def construct(self, topical_groups=None):

        self.pkg = nx.DiGraph()

        sag_entity_list   = set(list(itertools.chain.from_iterable(self.fellowship_list)))
        dipole_topic_list = set([t['topic']['id'] for t in self.attitude_list])

        """Append the Entity Relationships"""

        for e in self.sag.edges(data=True):

            e1 = self.int_to_node[e[0]]
            e2 = self.int_to_node[e[1]]
            w  = e[2]['weight']

            if w == 0: continue

            self.pkg.add_node(e1, type='Entity')
            self.pkg.add_node(e2, type='Entity')

            if w < 0: predicate='OpposeEE'
            else:     predicate='SupportEE'

            self.pkg.add_edge(e1, e2, weight=w, type='Relationship', predicate=predicate, label='NEGATIVE' if w < 0 else 'POSITIVE')
            self.pkg.add_edge(e2, e1, weight=w, type='Relationship', predicate=predicate, label='NEGATIVE' if w < 0 else 'POSITIVE')

        """Add the Entity-to-Topic Attitudes"""

        attitude_path_list            = []
        np_topics_dict, clean_np_dict = {}, {}

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'attitudes')):

            for p in files: attitude_path_list.append(os.path.join(root, p))

        for t in self.topic_dict:

            for np, c in zip(self.topic_dict[t]['noun_phrases'], self.topic_dict[t]['pre_processed']):

                clean_np_dict[np] = c

                if c not in np_topics_dict: np_topics_dict[c] = []

                np_topics_dict[c].append(t)

                np_topics_dict[c] = list(set(np_topics_dict[c]))

        entity_np_sentiment_attitudes = {}

        pool = Pool(multiprocessing.cpu_count() - 4)

        for result in tqdm(
                pool.imap_unordered(self.read_sentiment_attitudes, attitude_path_list),
                desc='Fetching Attitudes',
                total=len(attitude_path_list)
        ):
            for r in result:

                r = r['noun_phrase_attitudes']

                for p in r:

                    if p[1] not in clean_np_dict: break

                    if p[0] not in entity_np_sentiment_attitudes: entity_np_sentiment_attitudes[p[0]] = {}

                    c = clean_np_dict[p[1]]

                    if c not in entity_np_sentiment_attitudes[p[0]]: entity_np_sentiment_attitudes[p[0]][c] = []

                    entity_np_sentiment_attitudes[p[0]][c] += r[p]

        pool.close()
        pool.join()

        entity_topic_sentiment_attitude_dict = {}

        for e in entity_np_sentiment_attitudes:

            entity_topic_sentiment_attitude_dict[e] = {}

            for np in entity_np_sentiment_attitudes[e]:

                for t in np_topics_dict[np]:

                    if t not in entity_topic_sentiment_attitude_dict[e]: entity_topic_sentiment_attitude_dict[e][t] = []

                    entity_topic_sentiment_attitude_dict[e][t] += entity_np_sentiment_attitudes[e][np]

        attitude_values = []

        for e in entity_topic_sentiment_attitude_dict:

            for t in entity_topic_sentiment_attitude_dict[e]:

                attitude_values.append(numpy.mean(entity_topic_sentiment_attitude_dict[e][t]))

        bins = calculate_value_buckets(attitude_values)

        sentiment_mapping = {
            "NEGATIVE":  [(-1.000000, 0.000000)],
            "NEUTRAL":   [( 0.000000, 0.000001)],
            "POSITIVE":  [( 0.000001, 1.000000)]
        }

        for e in entity_topic_sentiment_attitude_dict:

            if e not in sag_entity_list: continue

            self.pkg.add_node(e, type='Entity')

            for t in entity_topic_sentiment_attitude_dict[e]:

                if t not in dipole_topic_list: continue

                self.pkg.add_node(t, type='Topic')

                v = numpy.median(entity_topic_sentiment_attitude_dict[e][t])
                s = convert_sentiment_attitude(v, sentiment_mapping)

                predicate = 'OpposeET' if s == 'NEGATIVE' else 'SupportET' if s == 'POSITIVE' else None

                if s == 'NEUTRAL': continue

                self.pkg.add_edge(e, t, type='Attitude', weight=v, label=s, predicate=predicate, observations=len([v for v in entity_topic_sentiment_attitude_dict[e][t]]))

        """Calculate Fellowship-to-Topic Attitudes"""

        fellowship_topic_attitude_dict = {}

        for i, f in enumerate(self.fellowship_list):

            if i not in fellowship_topic_attitude_dict: fellowship_topic_attitude_dict[i] = {}

            for e in f:

                if not e in entity_topic_sentiment_attitude_dict: continue

                for t in entity_topic_sentiment_attitude_dict[e]:

                    if t not in fellowship_topic_attitude_dict[i]: fellowship_topic_attitude_dict[i][t] = []

                    fellowship_topic_attitude_dict[i][t] += entity_topic_sentiment_attitude_dict[e][t]

        fellowship_topic_attitude_values = []

        for f in fellowship_topic_attitude_dict:

            for t in fellowship_topic_attitude_dict[f]:

                fellowship_topic_attitude_values += fellowship_topic_attitude_dict[f][t]

        bins = calculate_value_buckets(fellowship_topic_attitude_values)

        for f in fellowship_topic_attitude_dict:

            self.pkg.add_node(f'F{f}', type='Fellowship')

            for e in self.fellowship_list[f]:

                if e not in sag_entity_list: continue

                self.pkg.add_edge(e, f'F{f}', type='Member', predicate='MemberOf')

            for t in fellowship_topic_attitude_dict[f]:

                if t not in dipole_topic_list: continue

                v = numpy.mean(fellowship_topic_attitude_dict[f][t])
                s = convert_sentiment_attitude(v, sentiment_mapping)

                if s == 'NEUTRAL': continue

                predicate = 'OpposeFT' if s == 'NEGATIVE' else 'SupportFT' if s == 'POSITIVE' else None

                self.pkg.add_edge(f'F{f}', t, type='Collective_Attitude', predicate=predicate, weight=v, label=s)

        """Fellowship Dipole Definition """

        for d in self.dipole_list:

            self.pkg.add_node(f'D{d[0][0]}_{d[0][1]}', type='Dipole')

            self.pkg.add_edge(f'F{d[0][0]}', f'F{d[0][1]}', type='Conflict', predicate='Conflict', positive_edges=d[1]['pos'], negative_edges=d[1]['neg'], frustration=d[1]['f_g'])
            self.pkg.add_edge(f'F{d[0][0]}', f'D{d[0][0]}_{d[0][1]}', type='Part', predicate='PartOf')
            self.pkg.add_edge(f'F{d[0][1]}', f'D{d[0][0]}_{d[0][1]}', type='Part', predicate='PartOf')

        polarization_indices = [a['pi_res'] if 'pi_res' in a else a['pi'] for a in self.attitude_list]

        bins = calculate_value_buckets(polarization_indices)

        polarization_mapping      = {f'Polarization{i + 1}': [b] for i, b in enumerate(bins)}
        self.polarization_mapping = polarization_mapping

        #####################################################
        # print(json.dumps(polarization_mapping, indent=4)) #
        #####################################################

        for a in self.attitude_list:

            f1, f2 = a['dipole'][0], a['dipole'][1]

            d = f'D{f1}_{f2}'
            t = a['topic']['id']

            p = a['pi_res'] if 'pi_res' in a else a['pi']

            s = convert_sentiment_attitude(p, polarization_mapping)

            if s == 'NEUTRAL': s = 'Polarization0'

            self.pkg.add_edge(d, t, type='Polarization', weight=p, observations=len(a['atts_fi']) + len(a['atts_fj']), predicate=s, label=s)

    def get_node_by_type(self, type='Entity'): return [kv[0] for kv in dict(self.pkg.nodes(data=True)).items() if kv[1]['type'] == type]

    def get_positive_neighbors(self, node):

        neighbors = []
        for n1 in self._get_neighbors(node, attr_label='type', attr_value='Entity'):

            _ = self.pkg.get_edge_data(node, n1)
            if _['weight'] > 0: neighbors.append(n1)

        return neighbors

    def get_negative_neighbors(self, node):

        neighbors = []
        for n1 in self._get_neighbors(node, attr_label='type', attr_value='Entity'):

            _ = self.pkg.get_edge_data(node, n1)
            if _['weight'] < 0: neighbors.append(n1)

        return neighbors

    def get_entities(self):    return self.get_node_by_type(type='Entity')
    def get_fellowships(self): return self.get_node_by_type(type='Fellowship')
    def get_dipoles(self):     return self.get_node_by_type(type='Dipole')
    def get_topics(self):      return self.get_node_by_type(type='Topic')

    def _get_neighbors(self, node, attr_label=None, attr_value=None):

        u_pkg = self.pkg.to_undirected(as_view=True)

        neighbors = list(u_pkg.neighbors(node))

        if attr_label == None: return neighbors

        else: return [neighbor for neighbor in neighbors if u_pkg.nodes(data=True)[neighbor].get(attr_label) == attr_value]

    def get_fellowship_members(self, fellowship): return self._get_neighbors(fellowship, attr_label='type', attr_value='Entity')
    def get_fellowship_dipoles(self, fellowship): return self._get_neighbors(fellowship, attr_label='type', attr_value='Dipole')
    def get_dipole_fellowships(self, dipole):     return self._get_neighbors(dipole,     attr_label='type', attr_value='Fellowship')
    def get_entity_fellowship(self, entity):      return self._get_neighbors(entity,     attr_label='type', attr_value='Fellowship')

    def get_entity_topics(self, entity):

        topics = self.get_topics()
        topics = [t for t in self._get_neighbors(entity, attr_label='type', attr_value='Topic') if t in topics]

        return topics

    def get_dipole_topics(self, dipole):

        topics = self.get_topics()
        topics = [t for t in self._get_neighbors(dipole, attr_label='type', attr_value='Topic') if t in topics]

        return topics

    def get_dipole_topic_polarization(self, dipole, topics=None):

        if topics == None: topics = self.get_topics()

        topic_att_dict = {}

        dipole_topics = self.get_entity_topics(dipole)

        dipole_topics = list(set(dipole_topics).intersection(topics))

        for t in dipole_topics:

            topic_att_dict[t] = {
                'label': self.pkg.get_edge_data(dipole, t)['label'],
                'pi':    self.pkg.get_edge_data(dipole, t)['weight'],
                'obs':   self.pkg.get_edge_data(dipole, t)['observations']
            }

        return topic_att_dict

    def get_entity_topic_attitudes(self, entity, topics=None):

        if topics == None: topics = self.get_topics()

        topic_att_dict = {}

        entity_topics = self.get_entity_topics(entity)

        for t in entity_topics:

            if t not in topics: continue

            topic_att_dict[t] = self.pkg.get_edge_data(entity, t)['label']

        return topic_att_dict

    @staticmethod
    def read_sentiment_attitudes(path):

        with open(path, 'rb') as f: attitude_object = pickle.load(f)

        return attitude_object['attitudes']