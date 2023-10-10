import math, itertools, networkx as nx, pandas as pd, os, numpy

from tqdm import tqdm
from collections import Counter
from polarlib.prism.cohesiveness import cohesiveness
from polarlib.prism.polarization_knowledge_graph import PolarizationKnowledgeGraph

class EntityLevelPolarizationAnalyzer:

    @staticmethod
    def google_semantic_distance(e1, e2, G):

        if not G.has_node(e1): return 0.0
        if not G.has_node(e2): return 0.0

        in1 = set(G.neighbors(e1))
        in2 = set(G.neighbors(e2))

        union_n12 = in1.intersection(in2)

        log_u = 0.0
        if len(union_n12) > 0: log_u = math.log(len(union_n12))

        return (
                math.log(max(len(in1), len(in2))) - log_u
        ) / (
                math.log(G.number_of_nodes()) - math.log(min(len(in1), len(in2)))
        )

    @staticmethod
    def calculate_semantic_association(pkg: PolarizationKnowledgeGraph):

        entity_pair_association_dict = {}

        for e12 in tqdm(list(itertools.combinations(pkg.sag.nodes(), 2))):

            e1 = e12[0]
            e2 = e12[1]

            if e1 not in entity_pair_association_dict:     entity_pair_association_dict[e1] = {}
            if e2 not in entity_pair_association_dict[e1]: entity_pair_association_dict[e1][e2] = 0.0
            if e2 not in entity_pair_association_dict:     entity_pair_association_dict[e2] = {}
            if e1 not in entity_pair_association_dict[e1]: entity_pair_association_dict[e2][e1] = 0.0

            ###############################################
            # if not G.has_edge(e12[0], e12[1]): continue #
            ###############################################

            a = EntityLevelPolarizationAnalyzer.google_semantic_distance(e1, e2, pkg.sag)

            entity_pair_association_dict[e1][e2] = a
            entity_pair_association_dict[e2][e1] = a

        entity_association_dict  = {pkg.int_to_node[e]: sum(v.values()) for e, v in entity_pair_association_dict.items()}

        return entity_association_dict

    @staticmethod
    def signed_association(a_set, b_set, a_b_intersection, n):

        if len(a_b_intersection) == 0: return 0.0

        else: return (
                (math.log(max(len(a_set), len(b_set))) - math.log(len(a_b_intersection))) /
                (math.log(n) - math.log(min(len(a_set), len(b_set))))
        )

    @staticmethod
    def calculate_centrality(pkg: PolarizationKnowledgeGraph, func=nx.betweenness_centrality): return dict(func(pkg.pkg.subgraph(pkg.get_entities())))

    @staticmethod
    def calculate_signed_semantic_association(pkg: PolarizationKnowledgeGraph):
        signed_pair_semantic_association_dict = {}

        for e in pkg.sag.edges(data=True):

            a, b, relation = e[0], e[1], e[2]['weight']

            a_plus, a_minus = [], []
            b_plus, b_minus = [], []

            for n in pkg.sag.neighbors(a):

                weight = pkg.sag.get_edge_data(a, n)['weight']

                if   weight > 0.0: a_plus.append(n)
                elif weight < 0.0: a_minus.append(n)

            for n in pkg.sag.neighbors(b):

                weight = pkg.sag.get_edge_data(b, n)['weight']

                if   weight > 0.0: b_plus.append(n)
                elif weight < 0.0: b_minus.append(n)

            a_plus_b_plus   = list(set(a_plus).intersection(set(b_plus)))
            a_plus_b_minus  = list(set(a_plus).intersection(set(b_minus)))
            a_minus_b_plus  = list(set(a_minus).intersection(set(b_plus)))
            a_minus_b_minus = list(set(a_minus).intersection(set(b_minus)))

            a_plus_b_plus_association   = EntityLevelPolarizationAnalyzer.signed_association(a_plus,  b_plus,  a_plus_b_plus,   pkg.sag.number_of_nodes())
            a_plus_b_minus_association  = EntityLevelPolarizationAnalyzer.signed_association(a_plus,  b_minus, a_plus_b_minus,  pkg.sag.number_of_nodes())
            a_minus_b_plus_association  = EntityLevelPolarizationAnalyzer.signed_association(a_minus, b_plus,  a_minus_b_plus,  pkg.sag.number_of_nodes())
            a_minus_b_minus_association = EntityLevelPolarizationAnalyzer.signed_association(a_minus, b_minus, a_minus_b_minus, pkg.sag.number_of_nodes())

            if a not in signed_pair_semantic_association_dict: signed_pair_semantic_association_dict[a] = {}
            if b not in signed_pair_semantic_association_dict: signed_pair_semantic_association_dict[b] = {}

            if relation < 0.0:
                signed_pair_semantic_association_dict[a][b] = - a_plus_b_minus_association - a_minus_b_plus_association
                signed_pair_semantic_association_dict[b][a] = - a_plus_b_minus_association - a_minus_b_plus_association
            else:
                signed_pair_semantic_association_dict[a][b] =   a_plus_b_plus_association  + a_minus_b_minus_association
                signed_pair_semantic_association_dict[b][a] =   a_plus_b_plus_association  + a_minus_b_minus_association

        signed_semantic_association_dict = {pkg.int_to_node[e]: sum(v.values()) for e,v in signed_pair_semantic_association_dict.items()}

        return signed_semantic_association_dict

    @staticmethod
    def analyze(pkg: PolarizationKnowledgeGraph, verbose=False):

        semantic_association_dict        = EntityLevelPolarizationAnalyzer.calculate_semantic_association(pkg)

        if verbose: print('Calculated semantic association.')

        signed_semantic_association_dict = EntityLevelPolarizationAnalyzer.calculate_signed_semantic_association(pkg)

        if verbose: print('Calculated signed semantic association.')

        degree     = EntityLevelPolarizationAnalyzer.calculate_centrality(pkg, nx.degree_centrality)
        if verbose: print('Calculated degree centrality.')

        closeness  = EntityLevelPolarizationAnalyzer.calculate_centrality(pkg, nx.closeness_centrality)
        if verbose: print('Calculated closeness centrality.')

        # betweeness = EntityLevelPolarizationAnalyzer.calculate_centrality(pkg)
        # if verbose: print('Calculated betweeness centrality.')

        if verbose: print('Calculated centrality measures.')

        df = []

        for e in tqdm(pkg.get_entities(), desc='Populating Entity Metrics'):

            df.append({
                'entity':                e,
                'association':           semantic_association_dict[e] if e in semantic_association_dict else None,
                'signed_association':    signed_semantic_association_dict[e] if e in signed_semantic_association_dict else None,
                'degree_centrality':     degree[e],
                'closeness_centrality':  closeness[e],

                # 'betweeness_centrality': betweeness[e]
            })

        return pd.DataFrame.from_dict(df)

class GroupLevelPolarizationAnalyzer:

    @staticmethod
    def find_entity_ideology(pkg:PolarizationKnowledgeGraph, output_dir='./', download_flag=False, wlpa_flag=True):

        cohesiveness.DOWNLOAD_FLAG = download_flag

        entity_list = pkg.get_entities()

        entity_infobox_dict           = cohesiveness.fetch_entity_infoboxes(entity_list=entity_list, output_dir=os.path.join(output_dir, 'ideology'))
        party_list, entity_party_dict = cohesiveness.extract_entity_party(entity_infobox_dict)
        entity_affiliation_dict       = cohesiveness.get_entity_affiliations(entity_party_dict)

        if wlpa_flag:

            entity_affiliation_dict = cohesiveness.weighted_label_propagation_algorithm(
                pkg.sag,
                pkg.int_to_node,
                pkg.node_to_int,
                entity_affiliation_dict
            )

        return entity_affiliation_dict

    @staticmethod
    def calculate_ideological_cohesiveness(pkg:PolarizationKnowledgeGraph, output_dir='./', download_flag=False, wlpa_flag=True):

        ideological_cohesiveness_dict = {}

        entity_affiliation_dict = GroupLevelPolarizationAnalyzer.find_entity_ideology(pkg, output_dir, download_flag, wlpa_flag)

        for i, f in enumerate(pkg.fellowship_list):

            labels = [entity_affiliation_dict[e] for e in f]

            coh = cohesiveness.purity_score(labels, neutral_labels=['N'])

            ideological_cohesiveness_dict[f'F{i}'] = coh

        return ideological_cohesiveness_dict

    @staticmethod
    def calculate_attitudinal_cohesiveness(pkg:PolarizationKnowledgeGraph):

        attitudinal_cohesiveness_dict = {}

        for i, f in enumerate(pkg.fellowship_list):

            f_topic_att_dict = {}

            for e in f:

                for k,v in pkg.get_entity_topic_attitudes(e).items():

                    if k not in f_topic_att_dict: f_topic_att_dict[k] = []

                    f_topic_att_dict[k].append(v)

            attitudinal_cohesiveness_dict[f'F{i}'] = {}

            for t in f_topic_att_dict:

                labels = f_topic_att_dict[t]

                coh = cohesiveness.purity_score(labels, neutral_labels=['NEUTRAL'])
                c   = len(labels)
                c   = c / len(f)

                attitudinal_cohesiveness_dict[f'F{i}'][t] = {
                    'cohesiveness':        coh,
                    'member_size':         len(f),
                    'attitude_population': len(labels),
                    'member_ratio':        c
                }

        return attitudinal_cohesiveness_dict

    @staticmethod
    def analyze(pkg: PolarizationKnowledgeGraph, output_dir='./', download_flag=False, wlpa_flag=True):

        ideological_cohesiveness = GroupLevelPolarizationAnalyzer.calculate_ideological_cohesiveness(pkg, output_dir, download_flag, wlpa_flag)
        attitudinal_cohesiveness = GroupLevelPolarizationAnalyzer.calculate_attitudinal_cohesiveness(pkg)

        df1 = []
        df2 = []

        for i, f in enumerate(pkg.fellowship_list):

            f_label = f'F{i}'

            fg    = pkg.pkg.subgraph(f)
            edges = fg.edges(data=True)

            pos_edges = [(e[0], e[1], {'weight':  1}) for e in list(edges) if e[2]['weight'] > 0.0]
            neg_edges = [(e[0], e[1], {'weight': -1}) for e in list(edges) if e[2]['weight'] < 0.0]

            fgp = nx.Graph()
            fgn = nx.Graph()

            fgp.add_edges_from(pos_edges)
            fgn.add_edges_from(neg_edges)

            topical_cohesiveness = attitudinal_cohesiveness[f_label]

            for t in topical_cohesiveness:

                _ = topical_cohesiveness[t].copy()
                _['fellowship'] = f_label
                _['topic']      = t

                df2.append(_)

            attitudinal_cohesiveness_values = [v['cohesiveness'] for v in topical_cohesiveness.values()]

            df1.append({
                'fellowship':    f_label,
                'edges':         len(edges),
                'positive':      len(pos_edges),
                'negative':      len(neg_edges),
                'density':       nx.density(fg),
                'density_plus':  nx.density(fgp),
                'density_minus': nx.density(fgn),

                'ideological_cohesiveness':     ideological_cohesiveness[f_label],
                'attitudinal_cohesiveness_avg': numpy.mean(attitudinal_cohesiveness_values) if len(attitudinal_cohesiveness_values) > 0 else None,
                'attitudinal_cohesiveness_std': numpy.std(attitudinal_cohesiveness_values)  if len(attitudinal_cohesiveness_values) > 0 else None,
                'attitudinal_cohesiveness_max': max(attitudinal_cohesiveness_values)        if len(attitudinal_cohesiveness_values) > 0 else None,
                'attitudinal_cohesiveness_min': min(attitudinal_cohesiveness_values)        if len(attitudinal_cohesiveness_values) > 0 else None
            })

        return pd.DataFrame.from_dict(df1), pd.DataFrame.from_dict(df2)

class TopicLevelPolarizationAnalyzer:

    @staticmethod
    def calculate_local_topical_polarization(pkg: PolarizationKnowledgeGraph):

        local_polarization_dict = {}

        for d in tqdm(pkg.get_dipoles(), desc='Calculating Local Polarization'):
            local_polarization_dict[d] = pkg.get_dipole_topic_polarization(d)

        return local_polarization_dict

    @staticmethod
    def calculate_global_topical_polarization(pkg: PolarizationKnowledgeGraph):

        global_polarization_dict = {}

        for t in tqdm(pkg.get_topics(), desc='Calculating Global Polarization'):

            v = TopicLevelPolarizationAnalyzer._calculate_global_topical_polarization(pkg, t)
            global_polarization_dict[t] = v

        return global_polarization_dict

    @staticmethod
    def analyze(pkg:PolarizationKnowledgeGraph):

        local_polarization_dict = TopicLevelPolarizationAnalyzer.calculate_local_topical_polarization(pkg)
        # global_polarization_dict = TopicLevelPolarizationAnalyzer.calculate_global_topical_polarization(pkg)

        df1 = []
        df2 = []

        for d, tv in local_polarization_dict.items():

            for t, v in tv.items():
                v['dipole'] = d
                v['topic']  = t

                df1.append(v)

        df1 = pd.DataFrame.from_dict(df1)

        for d in df1.groupby(by='topic'):

            t    = d[0]
            dt   = d[1].shape[0]
            obst = d[1]['obs'].sum()
            mt   = numpy.median(d[1]['pi'])

            score = 0.00
            if dt > 0: score = (obst / dt) * mt

            df2.append({
                'topic': t,
                'dt':    dt,
                'obst':  obst,
                'mt':    mt,
                'score': score,
                'maxm':  max(d[1]['pi']) if dt > 0 else None,
                'minm':  min(d[1]['pi']) if dt > 0 else None,
                'stdm':  numpy.std(d[1]['pi']) if dt > 0 else None,
                'avgm':  numpy.mean(d[1]['pi']) if dt > 0 else None
            })

        return df1, pd.DataFrame.from_dict(df2)

    @staticmethod
    def _calculate_global_topical_polarization(pkg: PolarizationKnowledgeGraph, t):

        results = []

        for d in pkg.get_dipoles():

            d_polarization_dict = pkg.get_dipole_topic_polarization(d, topics=[t])

            if t in d_polarization_dict: results.append(d_polarization_dict[t])

        dt   = len(results)
        obst = sum([o['obs'] for o in results])
        mt   = numpy.median([o['pi'] for o in results])

        if dt == 0: return 0.0

        return {
            'score': (obst / dt) * mt,
            'dt':    dt,
            'obst':  obst,
            'mt':    mt
        }
