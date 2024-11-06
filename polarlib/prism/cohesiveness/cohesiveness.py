import wptools, time, os, pickle, itertools

from tqdm.notebook import tqdm
from html import unescape
from urllib.parse import urlparse, unquote
from collections import Counter

def get_dbpedia_infobox(wp_str):

    path    = urlparse(wp_str).path.split('resource/')[-1]
    path    = unquote(path)

    wp = wptools.page(path).get_parse(show=False)
    return wp.data

def get_infobox(wp_str):

    path    = urlparse(wp_str).path.split('wiki/')[-1]
    path    = unquote(path)

    wp = wptools.page(path).get_parse(show=False)
    return wp.data['infobox']

DOWNLOAD_FLAG = False

def fetch_entity_infoboxes(entity_list, output_dir='./cohesiveness/'):

    os.makedirs(output_dir, exist_ok=True)

    if DOWNLOAD_FLAG:

        entity_infobox_dict = {}

        for e in tqdm(entity_list):

            try: infobox = get_dbpedia_infobox(e)
            except Exception as ex:

                print(ex)
                continue

            entity_infobox_dict[e] = infobox.copy()

        with open(os.path.join(output_dir, 'entity_infobox_dict.pckl'), 'wb') as f: pickle.dump(entity_infobox_dict, f)

    else:

        with open(os.path.join(output_dir, 'entity_infobox_dict.pckl'), 'rb') as f: entity_infobox_dict = pickle.load(f)

    entity_infobox_dict = {k: v['infobox'] for k,v in entity_infobox_dict.items()}
    entity_infobox_dict = {k: v for k,v in entity_infobox_dict.items() if v}

    return entity_infobox_dict

import re
from bs4 import BeautifulSoup

def parse_wiki_template(tmpl):

    soup = BeautifulSoup(tmpl, features="lxml")
    return re.findall('\[\[(.*?)\]\]', soup.getText(' ').strip())

def get_ideological_information(wiki_infobox):

    name, ideology, position = '', '', ''

    if 'name'     in wiki_infobox:

        soup     = BeautifulSoup(wiki_infobox['name'])
        name     = soup.getText(' ').strip()

    if 'cohesiveness' in wiki_infobox: ideology = parse_wiki_template(wiki_infobox['cohesiveness'])
    if 'position' in wiki_infobox: position = parse_wiki_template(wiki_infobox['position'])

    ideology     = list(itertools.chain.from_iterable([i.split('|') for i in ideology]))
    ideology     = [i.strip() for i in ideology if len(i.strip()) > 0]

    str_response = name

    if isinstance(ideology, list) and len(ideology) > 0: str_response += ', ' + ', '.join(ideology)
    if isinstance(position, list) and len(position) > 0: str_response += ', ' + ', '.join(position)

    if str_response.startswith(', '): str_response = str_response[2:]

    return str_response

def extract_entity_party(entity_infobox_dict):

    party_list = [v['party'] for v in entity_infobox_dict.values() if 'party' in v]
    party_list = [parse_wiki_template(p) for p in party_list]
    party_list = [list(itertools.chain.from_iterable([i.split('|') for i in p])) for p in party_list]

    party_list = list(itertools.chain.from_iterable(party_list))

    entity_party_dict = {}

    for e, v in entity_infobox_dict.items():

        if 'party' not in v: continue

        p = v['party']

        p_list = parse_wiki_template(p)

        p_list = list(itertools.chain.from_iterable([i.split('|') for i in p_list]))
        p_list = [_ for _ in p_list if len(_.strip()) > 0]

        if len(p_list) > 0: entity_party_dict[e] = p_list.copy()

    return party_list, entity_party_dict

def get_party_ideologies(party_list, output_dir='./cohesiveness/'):

    if DOWNLOAD_FLAG:

        party_ideologies_dict = {}

        for p in set(party_list):

            _p = p.replace(' ', '_')

            try: p_infobox = get_infobox('https://en.wikipedia.org/wiki/' + _p)
            except Exception as ex:
                print(ex)
                continue

            p_ideologies = ''

            if p_infobox: p_ideologies = get_ideological_information(p_infobox)

            print('Party:     ', _p)
            print('Ideologies:', p_ideologies)
            print()

            if len(p_ideologies) > 0: party_ideologies_dict[p] = p_ideologies

        with open(os.path.join(output_dir, 'party_ideologies_dict.pckl'), 'wb') as f: pickle.dump(party_ideologies_dict, f)

    else:

        with open(os.path.join(output_dir, 'party_ideologies_dict.pckl'), 'rb') as f: party_ideologies_dict = pickle.load(f)

    return party_ideologies_dict

def get_entity_ideologies(entity_party_dict, party_ideologies_dict):

    entity_party_dbpedia_mapping = {}

    entity_ideology_dict = {}

    for e in entity_party_dict:

        for p in entity_party_dict[e]:

            if p not in party_ideologies_dict: continue

            entity_party_dbpedia_mapping[e] = p

            entity_ideology_dict[e] = party_ideologies_dict[p]

    return entity_party_dbpedia_mapping, entity_ideology_dict

def get_entity_affiliations(entity_party_dict):

    entity_affiliation_dict = {}

    democratic_entity_list = ['http://dbpedia.org/resource/Democratic_Party_(United_States)']
    republican_entity_list = ['http://dbpedia.org/resource/Republican_Party_(United_States)']

    for kv in entity_party_dict.items():

        if 'Democratic Party (United States)' in kv[1]: democratic_entity_list.append(kv[0])
        if 'Republican Party (United States)' in kv[1]: republican_entity_list.append(kv[0])

    for e in republican_entity_list:
        if not e in democratic_entity_list: entity_affiliation_dict[e] = 'R'

    for e in democratic_entity_list:
        if not e in republican_entity_list: entity_affiliation_dict[e] = 'D'

    return entity_affiliation_dict

def get_signed_neighbors(n1, G):
    n_plus, n_minus = [n1], []

    for n2 in G.neighbors(n1):
        if   G.get_edge_data(n1, n2)['weight'] > 0: n_plus.append(n2)
        elif G.get_edge_data(n1, n2)['weight'] < 0: n_minus.append(n2)

    return (n_plus, n_minus)

def signed_jaccard_similarity(n1, n2, G):
    n_plus_n1, n_minus_n1 = get_signed_neighbors(n1, G)
    n_plus_n2, n_minus_n2 = get_signed_neighbors(n2, G)

    s_plus = len(set(n_plus_n1).intersection(set(n_plus_n2))) + len(set(n_minus_n1).intersection(set(n_minus_n2)))
    s_minus = len(set(n_plus_n1).intersection(set(n_minus_n2))) + len(set(n_minus_n1).intersection(set(n_plus_n2)))

    return (s_plus - s_minus) / len(set(n_plus_n1 + n_minus_n1).union(set(n_plus_n2 + n_minus_n2)))

def weighted_label_propagation_algorithm(
        G,
        int_to_node,
        node_to_int,
        entity_affiliation_dict,
        n_steps = 5
):

    node_signed_similarity_dict = {}

    for n1 in tqdm(G.nodes(), desc='Calculating Signed Jacard Similarity'):
        if n1 not in node_signed_similarity_dict: node_signed_similarity_dict[n1] = {}
        for n2 in G.neighbors(n1):
            node_signed_similarity_dict[n1][n2] = signed_jaccard_similarity(n1, n2, G)

    node_label_dict = {k: entity_affiliation_dict[int_to_node[k]] if int_to_node[k] in entity_affiliation_dict else 'N' for k in G.nodes()}

    wlpa_label_dict_1 = {}
    wlpa_label_dict_2 = {}

    wlpa_flag = False
    convergence_flag = True
    convergence_count = 0

    while(convergence_flag):

        for n in tqdm(G.nodes(), desc='Applying WLPA'):

            if node_label_dict[n] != 'N': continue

            n_label, label_vote_dict = 'N', {l: [] for l in set(node_label_dict.values())}

            for k,v in node_signed_similarity_dict[n].items():
                if k in wlpa_label_dict_1: label_vote_dict[wlpa_label_dict_1[k]].append(v)
                else: label_vote_dict[node_label_dict[k]].append(v)

            label_vote_dict = {l: sum(v) for l,v in label_vote_dict.items()}

            majority_label = sorted(label_vote_dict.items(), key=lambda kv: kv[1], reverse=True)[0][0]

            n_label = majority_label

            wlpa_label_dict_1[n] = n_label

        sorted_node_id_list = sorted(set(list(wlpa_label_dict_1.keys()) + list(wlpa_label_dict_2.keys())))

        if [wlpa_label_dict_1[k] for k in sorted_node_id_list if k in wlpa_label_dict_1] == [wlpa_label_dict_2[k] for k in sorted_node_id_list if k in wlpa_label_dict_2]:
            convergence_count += 1
        else:
            convergence_count = 0

        if convergence_count == n_steps: convergence_flag = False

        wlpa_label_dict_2 = wlpa_label_dict_1.copy()

    wlpa_affiliation_dict = {k:v for k,v in node_label_dict.items()}
    wlpa_affiliation_dict = {k:wlpa_label_dict_1[k] if k in wlpa_label_dict_1 else v for k,v in wlpa_affiliation_dict.items()}

    wlpa_affiliation_dict = {int_to_node[k]: v for k,v in wlpa_affiliation_dict.items()}

    reps = [e for e in wlpa_affiliation_dict if wlpa_affiliation_dict[e] == 'R']
    dems = [e for e in wlpa_affiliation_dict if wlpa_affiliation_dict[e] == 'D']

    return wlpa_affiliation_dict

def purity_score(labels, neutral_labels=['N']):

    if all(n in neutral_labels for n in labels): return 0.00

    l      = Counter([l for l in labels if l not in neutral_labels]).most_common(1)[0][0]
    scores = [1 if l == _ else 0 for _ in labels]

    return (1 / len(labels)) * sum(scores)



def find_cohesive_fellowships(
        fellowships,
        entity_political_orientation_dict,
        purity_thr=0.5
):

    political_fellowship_list = []
    political_leaning_list    = []

    for i, f in enumerate(fellowships):

        l = Counter([
            entity_political_orientation_dict[e]
            if e in entity_political_orientation_dict else 'N' for e in f
        ])

        p_score = purity_score([
            entity_political_orientation_dict[e]
            if e in entity_political_orientation_dict else 'N' for e in f
        ])

        majority_label = Counter([
            entity_political_orientation_dict[e]
            if e in entity_political_orientation_dict else 'N' for e in f
        ]).most_common(1)[0][0]

        if p_score > purity_thr and majority_label != 'N':
            political_fellowship_list.append(f)
            political_leaning_list.append(majority_label)

    return political_fellowship_list, political_leaning_list