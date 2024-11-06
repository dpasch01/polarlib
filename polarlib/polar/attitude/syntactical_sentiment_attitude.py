import multiprocessing
import os, pickle, spacy, networkx as nx, itertools, pysentiment2 as ps

hiv4 = ps.HIV4()

from polarlib.utils.utils import *

from tqdm import tqdm

from polarlib.polar.attitude.mpqa import mpqa

from datasets.utils.logging import disable_progress_bar
from transformers import TrainingArguments, Trainer

disable_progress_bar()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class SyntacticalSentimentAttitudePipeline:
    """
    A class representing a syntactical sentiment attitude pipeline.

    This class initializes and manages various components for syntactical sentiment analysis.

    Args:
        output_dir (str): The directory to store output data.
        nlp: The natural language processing model or tool for text analysis.
        mpqa_path (str, optional): Path to the MPQA lexicon data. Default is None.

    Attributes:
        output_dir (str): The directory to store output data.
        nlp: The natural language processing model or tool for text analysis.
        mpqa_path (str): Path to the MPQA lexicon data.
        mpqa: An instance of the MPQA class for sentiment analysis using the MPQA lexicon.
        noun_phrase_path_list (list): A list of paths to noun phrase files.
    """

    def __init__(self, output_dir, nlp, mpqa_path=None):
        """
        Initialize the SyntacticalSentimentAttitudePipeline.

        Args:
            output_dir (str): The directory to store output data.
            nlp: The natural language processing model or tool for text analysis.
            mpqa_path (str, optional): Path to the MPQA lexicon data. Default is None.
        """

        self.output_dir = output_dir
        self.nlp = nlp
        self.mpqa_path = mpqa_path

        self.mpqa = mpqa(self.mpqa_path)
        self.mpqa.load_mpqa()

        self.noun_phrase_path_list = []

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'noun_phrases')):

            for p in files: self.noun_phrase_path_list.append(os.path.join(root, p))

    def find_longest_unique_path(self, graph, start, visited=None):
        if visited is None: visited = set()
        visited.add(start)

        longest_path = []
        for neighbor in graph[start]:
            if neighbor not in visited:
                path = self.find_longest_unique_path(graph, neighbor, visited.copy())
                if len(path) > len(longest_path): longest_path = path

        return [start] + longest_path

    def get_token_indices(self, doc, target):

        ###############################################
        # Find token indices that correspond to       #
        # the target string using character positions #
        ###############################################

        target_indices = []
        target_start_char = doc.text.find(target)
        target_end_char = target_start_char + len(target)

        for token in doc:
            if token.idx >= target_start_char and token.idx + len(token.text) <= target_end_char:
                target_indices.append(token.i)

        return target_indices

    def find_dependency_paths(self, text, source, target, verbose=False):
        """
        Find dependency paths between source and target entities in a given text.

        Args:
            text (str): The input text to analyze.
            source (list of tuples): List of tuples containing source entity information (text, from, to, label).
            target (list of tuples): List of tuples containing target entity information (text, from, to, label).
            verbose (bool, optional): If True, print verbose information. Default is False.

        Returns:
            list of lists: A list of lists representing the dependency paths between source and target entities.
        """

        document = self.nlp(text)

        graph = nx.Graph()

        for token in document:
            graph.add_node(token.i, text=token.text, lemma=token.lemma_, pos=token.pos_)
            if token.dep_ not in ['punct', 'ROOT']:
                graph.add_edge(token.head.i, token.i, dep=token.dep_)
                if verbose: print(token.head, token.dep_, token)

        source_token_indices = []
        target_token_indices = []

        for _ in source: source_token_indices += self.get_token_indices(document, _[0])
        for _ in target: target_token_indices += self.get_token_indices(document, _[0])

        path_list = []

        for pair in itertools.product(source_token_indices, target_token_indices):

            if graph.has_edge(pair[0], pair[1]):

                if graph.get_edge_data(pair[0], pair[1])['dep'] in ['ccomp', 'compound', 'conj']:

                    for n in graph.nodes():
                        path_list += list(nx.all_simple_paths(graph, pair[0], target=n))
                        path_list += list(nx.all_simple_paths(graph, pair[1], target=n))

            for p in nx.all_simple_paths(graph, pair[0], pair[1]): path_list.append(p)

        path_list = find_longest_unique_subsequences(sorted(path_list))
        path_list = [
            [document[i] for i in p if not document[i].is_stop and not i in source_token_indices + target_token_indices]
            for p in path_list]
        path_list = [p for p in path_list if len(p) > 0]

        return path_list

    def calculate_attitude(self, text, source, target):
        """
        Calculate the attitude expressed in the given text between the source and target entities.

        Args:
            text (str): The input text in which the attitude needs to be calculated.
            source (str): The source entity.
            target (str): The target entity.

        Returns:
            float: The calculated attitude score between the source and target entities.
                   A positive value indicates a positive attitude, a negative value indicates a negative attitude,
                   and 0.0 indicates a neutral attitude.
        """

        attitudes = []

        for p in self.find_dependency_paths(
                text=text,
                source=source,
                target=target,
                verbose=False
        ):

            ################################################################
            # attitude = hiv4.get_score([t.lemma_ for t in p])['Polarity'] #
            ################################################################

            attitude = self.mpqa.calculate_mpqa(p)

            if attitude == 0: continue

            attitudes.append(attitude)

        if len(attitudes) == 0:
            return 0.00
        else:
            return numpy.mean(attitudes)

    def extract_sentiment_attitude(self, path):
        """
        Extract sentiment and attitude information from a given file.

        Args:
            path (str): Path to the input file.

        Returns:
            bool: True if extraction is successful, False otherwise.
        """

        noun_phrase_entry = load_article(path)

        output_folder = os.path.join(self.output_dir, 'attitudes/' + path.split('/')[-2] + '/')
        output_file = os.path.join(output_folder, noun_phrase_entry['uid'] + '.pckl')

        if os.path.exists(output_file): return True

        for i in range(len(noun_phrase_entry['noun_phrases'])):

            noun_phrase_entry['noun_phrases'][i]['entity_attitudes'] = {}

            entity_reference_dict = {}

            s_from = noun_phrase_entry['noun_phrases'][i]['from']
            s = noun_phrase_entry['noun_phrases'][i]['sentence']

            if len(s) > 512: continue

            for e in noun_phrase_entry['noun_phrases'][i]['entities']:

                if e['title'] not in entity_reference_dict: entity_reference_dict[e['title']] = []
                entity_reference_dict[e['title']].append((e['text'], e['begin'] - s_from, e['end'] - s_from))

            for p in itertools.combinations(entity_reference_dict, 2):

                p_ = list(p)
                p_.sort()
                p_ = (p_[0], p_[1])

                if p_ not in noun_phrase_entry['noun_phrases'][i]['entity_attitudes']:
                    noun_phrase_entry['noun_phrases'][i]['entity_attitudes'][p_] = []

                e1 = entity_reference_dict[p[0]]
                e2 = entity_reference_dict[p[1]]

                for _ in itertools.product(e1, e2):
                    noun_phrase_entry['noun_phrases'][i]['entity_attitudes'][p_].append(
                        self.calculate_attitude(s, e1, e2))

        for i in range(len(noun_phrase_entry['noun_phrases'])):

            noun_phrase_entry['noun_phrases'][i]['noun_phrase_attitudes'] = {}

            entity_reference_dict = {}

            s_from = noun_phrase_entry['noun_phrases'][i]['from']
            s = noun_phrase_entry['noun_phrases'][i]['sentence']

            if len(s) > 512: continue

            for e in noun_phrase_entry['noun_phrases'][i]['entities']:

                if e['title'] not in entity_reference_dict: entity_reference_dict[e['title']] = []
                entity_reference_dict[e['title']].append((e['text'], e['begin'] - s_from, e['end'] - s_from))

            for p in itertools.product(entity_reference_dict, noun_phrase_entry['noun_phrases'][i]['noun_phrases']):

                p_ = (p[0], p[1]['ngram'])

                if p_ not in noun_phrase_entry['noun_phrases'][i]['noun_phrase_attitudes']:
                    noun_phrase_entry['noun_phrases'][i]['noun_phrase_attitudes'][p_] = []

                e1 = entity_reference_dict[p[0]]
                e1 = [list(_) + ['source'] for _ in e1]
                np2 = [[p[1]['ngram'], p[1]['from'] - s_from, p[1]['to'] - s_from, 'target']]

                for _ in itertools.product(e1, np2):
                    noun_phrase_entry['noun_phrases'][i]['noun_phrase_attitudes'][p_].append(
                        self.calculate_attitude(s, e1, np2))

        noun_phrase_entry['attitudes'] = noun_phrase_entry['noun_phrases'].copy()
        del noun_phrase_entry['noun_phrases']

        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
        with open(output_file, 'wb') as f: pickle.dump(noun_phrase_entry, f)

        return True

    def calculate_sentiment_attitudes(self):
        """
        Calculate sentiment attitudes for a list of noun phrase paths.
        """
        sentence_list = []

        for path in self.noun_phrase_path_list:
            _ = load_article(path)
            sentence_list += [t['sentence'] for t in _['noun_phrases']]

        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 32)

        for _ in tqdm(
                pool.imap_unordered(self.extract_sentiment_attitude, self.noun_phrase_path_list),
                desc='Extracting Attitudes',
                total=len(self.noun_phrase_path_list)
        ):
            pass

        pool.close()
        pool.join()

        """
        for i, path in enumerate(tqdm(self.noun_phrase_path_list)): 
            
            self.extract_sentiment_attitude(path)
        """

    def _replace_entity_indices(self, sentence, entities):
        """
        Replace entity indices in a sentence with special tokens.

        Args:
            sentence (str): Original sentence.
            entities (list): List of entity information.

        Returns:
            str: Sentence with replaced entity indices.
        """
        annotations = []
        sorted_entities = sorted(entities, key=lambda e: e[1])

        current_index = 0

        for entity in sorted_entities:
            start = entity[1]
            end = entity[2]
            label = entity[3]

            annotations.append(sentence[current_index:start])
            annotations.append(f"[{label.upper()}]")

            current_index = end

        annotations.append(sentence[current_index:])

        return ''.join(annotations)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    sentiment_attitude_pipeline = SyntacticalSentimentAttitudePipeline(
        output_dir="../example",
        nlp=nlp
    )

    sentiment_attitude_pipeline.calculate_sentiment_attitudes()
