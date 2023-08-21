import os, itertools, pandas as pd, pickle

from utilities import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from tqdm import tqdm
from datasets import Dataset
from scipy.special import softmax

from datasets.utils.logging import disable_progress_bar
from transformers import Trainer, TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy

disable_progress_bar()

class SentimentAttitudePipeline:

    def __init__(
            self,
            output_dir,
            model: AutoModelForSequenceClassification,
            tokenizer: AutoModelForSequenceClassification,
            training_args: TrainingArguments
    ):

        self.output_dir   = output_dir
        self.model        = model
        self.tokenizer    = tokenizer
        self.label_to_int = {'POSITIVE': 0, 'NEUTRAL': 1, 'NEGATIVE': 2}
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

        self.training_args = training_args

        self.model.cuda()

        self.trainer      =  Trainer(
            model         = self.model,
            args          = self.training_args,
            train_dataset = None,
            eval_dataset  = None
        )

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[SOURCE]", "[TARGET]"]})

        self.model_pipeline = TextClassificationPipeline(
            model             = self.model,
            tokenizer         = self.tokenizer,
            top_k             = None,
            device            = 0
        )

        self.noun_phrase_path_list = []

        for root, folders, files in os.walk(os.path.join(self.output_dir, 'noun_phrases')):

            for p in files: self.noun_phrase_path_list.append(os.path.join(root, p))

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def tokenize(self, batch):  return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

    def get_raw_attitude(self, anonymized):

        r = self.model_pipeline(
            anonymized,
            truncation=True,
            padding=True,
            max_length=512
        )

        r = r[0]

        r = [(self.int_to_label[int(_['label'].split('_')[1])], _['score']) for _ in r]

        r = list(sorted(r, key=lambda v: v[1], reverse=True))

        return r

    def attitude_inference(self, entity_sentence_list):

        entity_sentence_list = [t for t in entity_sentence_list if len(t['sentence'].strip()) > 5]

        if len(entity_sentence_list) == 0: return {}

        pair_list     = [e['pair'] for e in entity_sentence_list]
        sentence_list = [e['sentence'] for e in entity_sentence_list]

        df = []

        for e in entity_sentence_list:

            df.append({
                'e1':   e['pair'][0],
                'e2':   e['pair'][1],
                'text': e['sentence']
            })

        df = pd.DataFrame.from_dict(df)
        ds = Dataset.from_pandas(df)
        ds = ds.map(self.tokenize, batched=True, batch_size=None, )

        preds_output = self.trainer.predict(ds)
        probs_output = softmax(preds_output.predictions, axis=1)

        probs_output = [{
            self.int_to_label[k]: float(v[k])
            for k in range(len(v))
        } for v in probs_output]

        attitude_dict = {}

        for atts in zip(pair_list, probs_output):

            _ = (atts[0][0], atts[0][1])

            if _ not in attitude_dict: attitude_dict[_] = []

            attitude_dict[_].append(atts[1])

        return attitude_dict

    def extract_sentiment_attitude(self, path):

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

                entity_sentence_list.append({"pair": list(p_), "sentence": s_})

                e1 = entity_reference_dict[p[1]]
                e2 = entity_reference_dict[p[0]]
                e1 = [list(_) + ['source'] for _ in e1]
                e2 = [list(_) + ['target'] for _ in e2]

                entities = list(e1 + e2)
                s_ = self._replace_entity_indices(s, entities)

                entity_sentence_list.append({"pair": list(p_), "sentence": s_})

        noun_phrase_entry['entity_attitudes'] = self.attitude_inference(entity_sentence_list).copy()

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

                noun_phrase_sentence_list.append({"pair": list(p_), "sentence": s_})

        noun_phrase_entry['noun_phrase_attitudes'] = self.attitude_inference(noun_phrase_sentence_list).copy()

        noun_phrase_entry['attitudes'] = noun_phrase_entry['noun_phrases'].copy()
        del noun_phrase_entry['noun_phrases']

        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
        with open(output_file, 'wb') as f: pickle.dump(noun_phrase_entry, f)

        return True

    def calculate_sentiment_attitudes(self):

        sentence_list = []

        for path in self.noun_phrase_path_list:
            _              = load_article(path)
            sentence_list += [t['sentence'] for t in _['noun_phrases']]

        sentence_list = sorted(sentence_list, key=len, reverse=True)

        for i, path in enumerate(tqdm(self.noun_phrase_path_list)): self.extract_sentiment_attitude(path)

    def _replace_entity_indices(self, sentence, entities):
        sorted_entities = sorted(entities, key=lambda x: x[1], reverse=True)
        reconstructed_sentence = list(sentence)

        for entity, start, end, type in sorted_entities:
            source = sentence[start:end]
            target = f"[{type.upper()}]"

            reconstructed_sentence[start:end] = target

        return "".join(reconstructed_sentence)

if __name__ == "__main__":

    batch_size                  = 8
    gradient_accumulation_steps = 4
    logging_steps               = 10

    training_args = TrainingArguments(
        output_dir                  = '../.cache',
        weight_decay                = 0.02,
        learning_rate               = 1e-5,
        evaluation_strategy         = "epoch",
        save_strategy               = 'epoch',
        num_train_epochs            = 10,
        logging_steps               = logging_steps,
        load_best_model_at_end      = True,
        save_total_limit            = 2,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        disable_tqdm                = True
    )

    model_path = "/home/dpasch01/notebooks/Sentiment Attitude Classification/models/roberta-base-sentiment-attitude/pretrained"

    model_name = 'roberta-base'

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_path)

    sentiment_attitude_pipeline = SentimentAttitudePipeline(
        output_dir    = "../example",
        model         = model,
        tokenizer     = tokenizer,
        training_args = training_args
    )

    sentiment_attitude_pipeline.calculate_sentiment_attitudes()