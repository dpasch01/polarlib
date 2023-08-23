import os, itertools, ray, pickle

from polarlib.utils.utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, AutoConfig

from tqdm import tqdm

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class SentimentAttitudePipelineRay:

    def __init__(
            self,
            output_dir,
            model: AutoModelForSequenceClassification,
            tokenizer: AutoModelForSequenceClassification
    ):
        self.output_dir   = output_dir
        self.model        = model
        self.tokenizer    = tokenizer
        self.label_to_int = {'POSITIVE': 0, 'NEUTRAL': 1, 'NEGATIVE': 2}
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[SOURCE]", "[TARGET]"]})

        self.model_pipeline = TextClassificationPipeline(
            model             = self.model,
            tokenizer         = self.tokenizer,
            top_k             = None
        )

        ray.init(num_cpus=64, num_gpus=1, ignore_reinit_error=True)

        self.model_pipeline_id = ray.put(self.model_pipeline)

        self.noun_phrase_path_list = []

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'noun_phrases')):

            for p in files: self.noun_phrase_path_list.append(os.path.join(root, p))

    @ray.remote
    def ray_predict(pipeline, text_data): return pipeline(text_data)

    def attitude_inference_ray(self, entries):

        if len(entries) == 0: return []

        entry_batches = list(to_chunks(entries, 64))

        responses     = ray.get([self.ray_predict.remote(self.model_pipeline_id, [b['sentence'] for b in batch]) for batch in entry_batches])

        _entries = []

        for i, batch in enumerate(entry_batches):

            for j, e in enumerate(batch):

                e['attitude'] = responses[i][j].copy()

                _entries.append(e)

        return _entries.copy()

    def prepare_ray_inputs(self, path):

        entity_sentence_list      = []
        noun_phrase_sentence_list = []

        noun_phrase_entry = load_article(path)

        output_folder = os.path.join(self.output_dir, 'attitudes/' + path.split('/')[-2] + '/')
        output_file   = os.path.join(output_folder, noun_phrase_entry['uid'] + '.pckl')

        if os.path.exists(output_file): return True

        for i in range(len(noun_phrase_entry['noun_phrases'])):

            noun_phrase_entry['noun_phrases'][i]['entity_attitudes'] = {}

            entity_reference_dict = {}

            s_from = noun_phrase_entry['noun_phrases'][i]['from']
            s      = noun_phrase_entry['noun_phrases'][i]['sentence']

            if len(s) > 512: continue

            for e in noun_phrase_entry['noun_phrases'][i]['entities']:

                if e['title'] not in entity_reference_dict: entity_reference_dict[e['title']] = []
                entity_reference_dict[e['title']].append((e['text'], e['begin'] - s_from, e['end'] - s_from))

            for p in itertools.combinations(entity_reference_dict, 2):

                p_ = list(p)
                p_.sort()
                p_ = (p_[0], p_[1])

                if p_ not in noun_phrase_entry['noun_phrases'][i]['entity_attitudes']: noun_phrase_entry['noun_phrases'][i]['entity_attitudes'][p_] = []

                e1 = entity_reference_dict[p[0]]
                e2 = entity_reference_dict[p[1]]
                e1 = [list(_) + ['source'] for _ in e1]
                e2 = [list(_) + ['target'] for _ in e2]

                entities = list(e1 + e2)
                s_ = self._replace_entity_indices(s, entities)

                entity_sentence_list.append({"pair": list(p_), "sentence": s_, "type": "entity"})

                e1 = entity_reference_dict[p[1]]
                e2 = entity_reference_dict[p[0]]
                e1 = [list(_) + ['source'] for _ in e1]
                e2 = [list(_) + ['target'] for _ in e2]

                entities = list(e1 + e2)
                s_ = self._replace_entity_indices(s, entities)

                entity_sentence_list.append({"pair": list(p_), "sentence": s_, "type": "entity"})

        for i in range(len(noun_phrase_entry['noun_phrases'])):

            noun_phrase_entry['noun_phrases'][i]['noun_phrase_attitudes'] = {}

            entity_reference_dict = {}

            s_from = noun_phrase_entry['noun_phrases'][i]['from']
            s      = noun_phrase_entry['noun_phrases'][i]['sentence']

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

                entries = list(e1 + np2)
                s_ = self._replace_entity_indices(s, entries)

                noun_phrase_sentence_list.append({"pair": list(p_), "sentence": s_, "type": "noun_phrase"})

        return entity_sentence_list + noun_phrase_sentence_list

    def _(self, ray_input, id):

        ray_input = [i for i in ray_input if i and i['sentence']]

        response = self.attitude_inference_ray(ray_input)

        entity_predictions      = {}
        noun_phrase_predictions = {}

        for _ in response:

            if _['type'] == 'entity':

                _['pair'] = (_['pair'][0], _['pair'][1])

                if _['pair'] not in entity_predictions: entity_predictions[_['pair']] = []

                entity_predictions[_['pair']].append({t['label']: t['score'] for t in _['attitude']})

            if _['type'] == 'noun_phrase':

                _['pair'] = (_['pair'][0], _['pair'][1])

                if _['pair'] not in noun_phrase_predictions: noun_phrase_predictions[_['pair']] = []

                noun_phrase_predictions[_['pair']].append({t['label']: t['score'] for t in _['attitude']})

        output_folder = os.path.join(self.output_dir, "attitudes/")
        e_output_file = os.path.join(output_folder, str(id) + '.entity.json')
        n_output_file = os.path.join(output_folder, str(id) + '.noun_phrase.json')

        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
        with open(e_output_file, 'wb') as f: pickle.dump(entity_predictions, f)
        with open(n_output_file, 'wb') as f: pickle.dump(noun_phrase_predictions, f)

        return True

    def calculate_sentiment_attitudes(self):

        batches = list(to_chunks(self.noun_phrase_path_list, 16))

        for i, batch in enumerate(tqdm(batches)):

            ray_inputs = []

            for p in tqdm(batch): ray_inputs += self.prepare_ray_inputs(p)

            self._(ray_inputs, i)

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
            end   = entity[2]
            label = entity[3]

            annotations.append(sentence[current_index:start])
            annotations.append(f"[{label.upper()}]")

            current_index = end

        annotations.append(sentence[current_index:])

        return ''.join(annotations)

if __name__ == "__main__":

    model_path = "/home/dpasch01/notebooks/Sentiment Attitude Classification/models/roberta-base-sentiment-attitude/pretrained"

    model_name = 'roberta-base'

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    config     = AutoConfig.from_pretrained(model_path)

    label_to_int = {'POSITIVE': 0, 'NEUTRAL': 1, 'NEGATIVE': 2}
    int_to_label = {v:k for k,v in label_to_int.items()}

    config.id2label = {int(k):v for k,v in int_to_label.items()}
    config.label2id = {k: int(v) for k,v in label_to_int.items()}

    model      = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    sentiment_attitude_pipeline = SentimentAttitudePipelineRay(
        output_dir    = "/home/dpasch01/temp",
        model         = model,
        tokenizer     = tokenizer
    )

    sentiment_attitude_pipeline.calculate_sentiment_attitudes()