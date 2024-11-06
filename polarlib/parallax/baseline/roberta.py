import evaluate, torch, numpy, pandas as pd, datasets, os, matplotlib.pyplot as plt

from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from datasets import Dataset, DatasetDict

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

class RoBERTaBaseline:

    def __init__(self, training_set, test_set):

        model_name = "roberta-base"

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.training_set = training_set
        self.test_set     = test_set

        self.metrics = evaluate.combine(["accuracy", "f1", "precision", "recall", "roc_auc"])

        self.y_train = list(training_set.label.values)
        self.X_train = list(training_set.text.values)
        self.y_test  = list(test_set.label.values)
        self.X_test  = list(test_set.text.values)

        self.model_name = model_name

        df_train = pd.DataFrame({'text': self.X_train, 'label': self.y_train})
        df_test  = pd.DataFrame({'text': self.X_test,  'label': self.y_test})

        ds_train = Dataset.from_pandas(df_train)
        ds_test  = Dataset.from_pandas(df_test)

        ds = DatasetDict()

        ds['train']        = ds_train
        ds['validation']   = ds_test

        self.dataset         = ds
        self.tokenizer       = RobertaTokenizerFast.from_pretrained(self.model_name, do_lower_case = True, truncation=True, padding=True, max_length=512)
        self.encoded_dataset = None
        self.trainer         = None

        self.config = RobertaConfig.from_pretrained(self.model_name, output_hidden_states=True, num_labels=len(set(self.y_train + self.y_test)))
        self.model  = RobertaForSequenceClassification.from_pretrained(self.model_name, config=self.config)

    def fit(self, output_dir='./roberta_parallax_baseline', batch_size=4, gradient_accumulation_steps=4):

        def model_init(): return RobertaForSequenceClassification.from_pretrained(self.model_name, config=self.config)

        optimizer = Adafactor(
            self.model.parameters(),
            lr              = 1e-3,
            eps             = (1e-30, 1e-3),
            clip_threshold  = 1.0,
            decay_rate      = -0.8,
            beta1           = None,
            weight_decay    = 0.0,
            relative_step   = False,
            scale_parameter = False,
            warmup_init     = False,
        )

        lr_scheduler = AdafactorSchedule(optimizer)

        batch_size                  = batch_size
        gradient_accumulation_steps = gradient_accumulation_steps

        logging_steps = len(self.encoded_dataset["train"]) // (batch_size * gradient_accumulation_steps)

        training_args = TrainingArguments(
            output_dir             = output_dir,
            weight_decay           = 0.02,
            learning_rate          = 2e-5,
            evaluation_strategy    = "epoch",
            save_strategy          = 'epoch',
            num_train_epochs       = 10,
            logging_steps          = logging_steps,
            load_best_model_at_end = True,
            save_total_limit       = 4,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps
        )

        self.trainer = Trainer(
            model_init      = model_init,
            args            = training_args,
            train_dataset   = self.encoded_dataset["train"],
            eval_dataset    = self.encoded_dataset["validation"],
            callbacks       = [EarlyStoppingCallback(early_stopping_patience = 3)],

            ###############################################
            # model_init      = model_init                #
            # model           = model                     #
            # compute_metrics = compute_metrics           #
            #                                             #
            # optimizers      = (optimizer, lr_scheduler) #
            ###############################################
        )

        self.trainer.train()
        self.model = self.trainer.model

        return self.trainer

    def save_model(self, path='./roberta_parallax_baseline_model'): self.model.save_pretrained(path, save_config=True)

    def get_raw_embeddings(self, text):
        input_ids     = torch.tensor(self.tokenizer.encode("[CLS] {}".format(text), truncation=True, padding=True, max_length=512)).unsqueeze(0).to(self.device)
        outputs       = self.model(input_ids)
        logits        = outputs.logits
        hidden_states = outputs.hidden_states

        return hidden_states[-1][:, 0, :].detach().cpu().numpy()[0]

    def evaluate(self):

        preds_output = self.trainer.predict(self.encoded_dataset["validation"])
        y_preds      = numpy.argmax(preds_output.predictions[0], axis=1)

        def plot_confusion_matrix(y_preds, y_true, labels):

            cm      = confusion_matrix(y_true, y_preds, normalize="true")
            fig, ax = plt.subplots(figsize=(6, 6))
            disp    = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

            plt.rcParams.update({'font.size': 8})

            disp.plot(
                cmap          = "Blues",
                values_format = ".2f",
                ax            = ax,
                colorbar      = False
            )

            plt.xticks(rotation=-90)

            plt.title("Normalized Confusion Matrix")
            plt.show()

        plot_confusion_matrix(
            y_preds,
            self.y_test,
            ['Reliable', 'Unreliable']
        )

        print(f"F1 Score : {f1_score(y_preds, self.y_test)}")
        print(f"Accuracy : {accuracy_score(y_preds, self.y_test)}")
        print(f"Precision: {precision_score(y_preds, self.y_test)}")
        print(f"Recall   : {recall_score(y_preds, self.y_test)}")
        print(f"ROC AUC  : {roc_auc_score(y_preds, self.y_test)}")
        print()
        print(classification_report(y_preds, self.y_test))
        print()

    def encode_dataset(self): self.encoded_dataset = self.dataset.map(self.tokenize, batched = True, batch_size = None)

    def tokenize(self, batch): return self.tokenizer(batch["text"], padding = True, truncation = True, max_length = 512)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions    = numpy.argmax(logits, axis=-1)

        return self.metrics.compute(predictions=predictions, references=labels, prediction_scores=predictions)




