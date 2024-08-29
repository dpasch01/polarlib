import json
import os
import optuna

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

import matplotlib.pyplot as plt, seaborn as sns, numpy, torch

from polarlib.parallax.pkg_embeddings import PKGEmbeddings
from polarlib.parallax.pkgnn.pkg_dataset import PKGDataset

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F, pandas as pd

from tqdm import tqdm
from torch.nn import Linear, BatchNorm1d, ModuleList, CrossEntropyLoss
from torch_geometric.nn import TransformerConv, TopKPooling, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader

import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import StratifiedShuffleSplit

from torch.nn import Linear
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

from torch.nn.utils.rnn import pad_sequence

def stratified_split(dataset, test_ratio=0.2, random_state=11):
    # Extract labels
    labels = [sample.y.item() for sample in dataset]

    # Create stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_idx, test_idx = next(sss.split(labels, labels))

    # Create train and test datasets
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]

    return train_dataset, test_dataset

def custom_collate(batch):

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_sequences, labels

class TrainingSession(object):

    @staticmethod
    def fit(params, dataset:PKGDataset, train_dataset:PKGDataset, test_dataset:PKGDataset, epochs=100, article_feature_dict=None):

        history = []

        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], collate_fn=custom_collate, shuffle=True, drop_last=True)
        test_loader  = DataLoader(test_dataset, batch_size=params["batch_size"], collate_fn=custom_collate, shuffle=False, drop_last=True)

        # Loading the model
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = PKGNN(dataset=dataset, model_params=model_params, article_feature_dict=article_feature_dict)
        model = model.to(DEVICE)

        print()

        print(f"Number of parameters: {TrainingSession.count_parameters(model)}")

        # weight    = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(DEVICE)
        # loss_fn   = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

        loss_fn     = CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr           =params["learning_rate"],
            momentum     =params["sgd_momentum"],
            weight_decay =params["weight_decay"]
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        best_loss              = 1000
        early_stopping_counter = 0

        for epoch in range(epochs):

            if early_stopping_counter <= 10:

                model.train()

                loss = TrainingSession.train_epoch(epoch, model, train_loader, optimizer, loss_fn)

                train_loss = loss[0]
                train_preds, train_labels = loss[1], loss[2]

                loss = train_loss

                print()
                print(f"Epoch {epoch} | Train Loss {loss}")

                model.eval()

                if epoch % 5 == 0:

                    loss = TrainingSession.test(epoch, model, test_loader, loss_fn)

                    test_loss = loss[0]

                    test_preds, test_labels = loss[1], loss[2]

                    loss = test_loss

                    history.append([train_loss, test_loss, {
                        'train_preds':  train_preds,
                        'train_labels': train_labels,
                        'test_preds':   test_preds,
                        'test_labels':  test_labels
                    }])

                    print()
                    print(f"Epoch {epoch} | Test Loss {loss}")

                    if float(loss) < best_loss:
                        best_loss = loss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()

            else:
                print()
                print("Early stopping due to no improvement.")
                return [history, best_loss, model]

        print(f"Finishing training with best test loss: {best_loss}")

        return [history, best_loss, model]

    @staticmethod
    def test(epoch, model, test_loader, loss_fn):

        y_true = []
        y_pred = []
        running_loss  = 0.0
        step          = 0

        for batch in test_loader:

            batch.to(DEVICE)

            pred    = model(batch)
            y_pred += list(pred.argmax(dim=1).cpu().detach().numpy())
            y_true += list(batch.y.cpu().detach().numpy())

            loss    = loss_fn(pred, batch.y)

            running_loss += loss.item()
            step         += 1

        """
        all_preds     = []
        all_preds_raw = []
        all_labels    = []
        running_loss  = 0.0
        step          = 0

        for batch in test_loader:

            batch.to(DEVICE)

            pred = model(batch)

            loss = loss_fn(pred, batch.y)

            running_loss += loss.item()
            step         += 1

            pred = pred.argmax(dim=1)

            all_preds.append(pred.cpu().detach().numpy())
            all_labels.append(batch.y.cpu().detach().numpy())

        all_preds = numpy.concatenate(all_preds).ravel()
        all_labels = numpy.concatenate(all_labels).ravel()

        TrainingSession.calculate_metrics(all_preds, all_labels, epoch, "test")

        return [running_loss / step, all_preds, all_labels]
        """
        if step == 0: return [running_loss, y_pred, y_true]

        return [running_loss / step, y_pred, y_true]

    @staticmethod
    def train_epoch(epoch, model, train_loader, optimizer, loss_fn, feature_dict = None):

        y_true = []
        y_pred = []

        running_loss, step = 0.0, 0

        """
        all_preds, all_labels = [], []
        """

        for batch in tqdm(train_loader, desc='Training Epoch {}'.format(epoch + 1)):

            batch.to(DEVICE)

            optimizer.zero_grad()

            pred = model(batch)

            loss = loss_fn(pred, batch.y)

            if feature_dict:

                """
                
                ##################
                # Regularization #
                ##################
                
                embeddings = torch.FloatTensor([feature_dict[p] for p in batch.path])
                l2_reg     = torch.norm(embeddings, p=2)
                reg_loss   = 0.01 * l2_reg
                loss       = loss + reg_loss
                """

                #####################
                # Attention Penalty #
                #####################

                loss += model.attention_penalty * 1e-2

            loss.backward()

            max_grad_norm = 20.0
            clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            running_loss += loss.item()
            step         += 1

            y_pred += list(pred.argmax(dim=1).cpu().detach().numpy())
            y_true += list(batch.y.cpu().detach().numpy())

        """
            pred = pred.argmax(dim=1)

            all_preds.append(pred.cpu().detach().numpy())
            all_labels.append(batch.y.cpu().detach().numpy())

        all_preds  = numpy.concatenate(all_preds).ravel()
        all_labels = numpy.concatenate(all_labels).ravel()
        
        TrainingSession.calculate_metrics(all_preds, all_labels, epoch, "train")

        return [running_loss / step, all_preds, all_labels]
        """

        return [running_loss / step, y_pred, y_true]

    @staticmethod
    def log_conf_matrix(y_pred, y_true, epoch, output):

        cm      = confusion_matrix(y_pred, y_true)
        classes = ["0", "1"]

        df_cfm = pd.DataFrame(cm, index = classes, columns = classes)

        plt.figure(figsize = (10,7))
        cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')

        plt.savefig(output)

    @staticmethod
    def calculate_metrics(y_pred, y_true, epoch, type):
        print()
        print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
        print(f"F1 Score: {f1_score(y_pred, y_true)}")
        print(f"Accuracy: {accuracy_score(y_pred, y_true)}")

        prec = precision_score(y_pred, y_true)
        rec = recall_score(y_pred, y_true)

        print(f"Precision: {prec}")
        print(f"Recall: {rec}")

        try:
            roc = roc_auc_score(y_pred, y_true)
            print(f"ROC AUC: {roc}")
        except: print(f"ROC AUC: N/A")

        print()

    @staticmethod
    def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def hyperparameter_tuning(trials=10, dataset=None, article_feature_dict=None, input_train_dataset=None, input_test_dataset=None, isolate='graph'):

        def objective(trial):

            params = {
                "weight_decay":          trial.suggest_categorical("weight_decay",          [0.01, 0.02, 0.03]),
                "model_attention_heads": trial.suggest_categorical("model_attention_heads", [1, 2, 3]),
                "pos_weight" :           1.0,
                "batch_size":            trial.suggest_categorical("batch_size",            [16, 32, 64]),
                "model_dropout_rate":    trial.suggest_categorical("model_dropout_rate",    [0.1, 0.2, 0.3]),
                "scheduler_gamma":       trial.suggest_categorical("scheduler_gamma",       [0.1, 0.2, 0.8, 0.9]),
                "sgd_momentum":          trial.suggest_categorical("sgd_momentum",          [0.2, 0.5, 0.8]),

                "scheduler":             trial.suggest_categorical("scheduler",             [1, 2, 3, 4, 5]),

                "model_layers":          trial.suggest_categorical("model_layers",          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "learning_rate":         trial.suggest_categorical("learning_rate",         [0.05, 0.02, 0.002, 0.0002, 0.01, 0.001]),
                "model_embedding_size":  trial.suggest_categorical("model_embedding_size",  [32, 64, 128, 256]),
                "model_dense_neurons":   trial.suggest_categorical("model_dense_neurons",   [128, 256, 64]),

                "model_top_k_ratio":     trial.suggest_categorical("model_top_k_ratio",   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                "model_top_k_every_n":   trial.suggest_categorical("model_top_k_every_n", [1, 2, 3]),

                "model_isolate":                   isolate,
                "model_feature_transform_flag":    trial.suggest_categorical("model_feature_transform_flag",    [True, False]),
                "model_concat_normalization_flag": trial.suggest_categorical("model_concat_normalization_flag", [True, False]),
                "model_feature_attention_flag":    trial.suggest_categorical("model_feature_attention_flag",    [True, False])
            }

            if isolate == 'graph': params['model_feature_transform_flag'] = False

            return calibrate(params)

        def calibrate(params):

            print(json.dumps(params, indent=4))

            if not input_train_dataset:

                print('Training / Testing datasets not found.')
                train_dataset, test_dataset = stratified_split(dataset, test_ratio=0.33, random_state=11)

            else:

                train_dataset = input_train_dataset
                test_dataset  = input_test_dataset

            if article_feature_dict:

                for i,d in enumerate(train_dataset):

                    if d.path not in article_feature_dict: train_dataset[i] = None

                train_dataset = [d for d in train_dataset if d]

                for i,d in enumerate(test_dataset):

                    if d.path not in article_feature_dict: test_dataset[i] = None

                test_dataset = [d for d in test_dataset if d]

            params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], collate_fn=custom_collate, shuffle=True, drop_last=True)
            test_loader  = DataLoader(test_dataset, batch_size=params["batch_size"], collate_fn=custom_collate, shuffle=False, drop_last=True)

            model_params = {k: v for k, v in params.items() if k.startswith("model_")}
            model        = PKGNN(dataset=dataset, model_params=model_params, article_feature_dict=article_feature_dict)
            model        = model.to(DEVICE)

            loss_fn   = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr           = params["learning_rate"],
                momentum     = params["sgd_momentum"],
                weight_decay = params["weight_decay"]
            )

            scheduler_list = [
                torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"]),
                torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=100, anneal_strategy='linear'),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001),
                torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=200, mode='triangular')
            ]

            print(scheduler_list)

            scheduler = scheduler_list[params["scheduler"] - 1]

            best_precision         = 0
            best_loss              = 1000
            early_stopping_counter = 0

            for epoch in range(100):

                if early_stopping_counter <= 10:

                    model.train()

                    loss = TrainingSession.train_epoch(epoch, model, train_loader, optimizer, loss_fn, feature_dict=article_feature_dict)

                    pred = loss[1]
                    true = loss[2]
                    loss = loss[0]

                    model.eval()

                    if epoch % 5 == 0:

                        loss = TrainingSession.test(epoch, model, test_loader, loss_fn)
                        pred = loss[1]
                        true = loss[2]
                        loss = loss[0]

                        prec = 0
                        try: prec = roc_auc_score(pred, true)
                        except: pass

                        if float(prec) > best_precision:
                            best_precision         = prec
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1

                    scheduler.step()

                else:

                    print(classification_report(y_true=true, y_pred=pred))

                    return best_precision

            print(classification_report(y_true=true, y_pred=pred))

            return best_precision

        opt_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

        opt_study.optimize(objective, n_trials=trials)

        print('Optuna Best:', opt_study.best_value)
        print()
        print(json.dumps(opt_study.best_params, indent=4))

        return opt_study

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(y_true, y_pred, average='weighted'):

    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {prec}")
    print(f"Recall:    {rec}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average=average, zero_division=0)}")

    try:
        roc = roc_auc_score(y_true, y_pred, average=average)
        print(f"ROC AUC:   {roc}")
    except: print(f"ROC AUC: N/A")

    print()

class PKGNNv2(torch.nn.Module):

    def __init__(self, dataset, model_params, ignore_features=False, article_feature_dict=None):

        super().__init__()

        self.clf = LogisticRegression(solver='liblinear', max_iter=1000)

        self.feature_dict = article_feature_dict

        if self.feature_dict: model_params['model_additional_features'] = len(next(iter(article_feature_dict.values())))

        if 'model_additional_features' not in model_params: model_params['model_additional_features'] = 0

        embedding_size     = model_params["model_embedding_size"]
        n_heads            = model_params["model_attention_heads"]
        dropout_rate       = model_params["model_dropout_rate"]
        dense_neurons      = model_params["model_dense_neurons"]
        edge_dim           = model_params["model_edge_dim"]
        top_k_ratio        = model_params["model_top_k_ratio"]

        self.attention_penalty_threshold = model_params["model_attention_penalty_threshold"]

        self.top_k_every_n   = model_params["model_top_k_every_n"]
        self.ignore_features = ignore_features
        self.n_layers        = model_params["model_layers"]
        self.isolate         = model_params["model_isolate"]

        self.feature_dimensions        = model_params['model_additional_features']
        self.feature_transform_flag    = model_params['model_feature_transform_flag']
        self.concat_normalization_flag = model_params['model_concat_normalization_flag']
        self.feature_attention_flag    = model_params["model_feature_attention_flag"]
        self.top_pooling_flag          = model_params["model_top_pooling_flag"]

        self.pooling_layers = ModuleList([])
        self.conv_layers    = ModuleList([])
        self.transf_layers  = ModuleList([])
        self.bn_layers      = ModuleList([])

        self.conv1 = TransformerConv(
            dataset.num_node_features,
            embedding_size,
            heads    = n_heads,
            dropout  = dropout_rate,
            edge_dim = edge_dim,
            beta     = True
        )

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1     = BatchNorm1d(embedding_size)

        self.attention_penalty = 0.0

        for i in range(self.n_layers):

            self.conv_layers.append(
                TransformerConv(
                    embedding_size,
                    embedding_size,
                    heads    = n_heads,
                    dropout  = dropout_rate,
                    edge_dim = edge_dim,
                    beta     = True
                )
            )

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

            if self.top_pooling_flag and i % self.top_k_every_n == 0:

                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        if self.feature_transform_flag:

            self.feature_transform  = torch.nn.Linear(self.feature_dimensions, embedding_size)
            self.feature_dimensions = embedding_size

        if self.feature_attention_flag:

            self.attention_linear  = torch.nn.Linear(embedding_size * 2 + self.feature_dimensions, 1)
            self.attention_softmax = torch.nn.Softmax(dim=1)

        if self.isolate == 'disable':

            self.concat_bn = BatchNorm1d(embedding_size * 2 + self.feature_dimensions)
            self.linear1   = Linear(embedding_size * 2 + self.feature_dimensions, dense_neurons)

        elif self.isolate == 'features':

            self.concat_bn = BatchNorm1d(self.feature_dimensions)
            self.linear1   = Linear(self.feature_dimensions, dense_neurons)

        elif self.isolate == 'graph':

            self.concat_bn = BatchNorm1d(embedding_size * 2)
            self.linear1   = Linear(embedding_size * 2, dense_neurons)

        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))
        self.linear3 = Linear(int(dense_neurons/2), dataset.num_classes)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.ignore_features:
            x         = torch.ones_like(x)
            edge_attr = torch.ones_like(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(self.transf1(x))

        x = self.bn1(x)

        global_representation = []

        for i in range(self.n_layers):

            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = F.leaky_relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

            if self.top_pooling_flag and (i % self.top_k_every_n == 0 or i + 1 == self.n_layers):

                x , edge_index, edge_attr, batch, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x,
                    edge_index,
                    edge_attr,
                    batch
                )

                global_representation.append(torch.cat([
                    gmp(x, batch),
                    gap(x, batch)
                ], dim=1))

            else:

                global_representation.append(torch.cat([
                    gmp(x, batch),
                    gap(x, batch)
                ], dim=1))

        x = sum(global_representation)

        if self.feature_dimensions > 0:

            additional_features = torch.FloatTensor([self.feature_dict[p] for p in data.path]).cuda()

            if self.feature_transform_flag:

                additional_features = self.feature_transform(additional_features)

            if self.isolate == 'disable':
                x                   = torch.cat((x, additional_features), -1)

                if self.feature_attention_flag:

                    attention_weights   = self.attention_softmax(self.attention_linear(x))
                    x                   = attention_weights * x

                    feature_weights     = attention_weights[:, -self.feature_dimensions:]
                    mean_weight         = feature_weights.mean()
                    penalty_threshold   = self.attention_penalty_threshold
                    penalty             = F.leaky_relu(mean_weight - penalty_threshold)

                    self.attention_penalty = penalty

            if self.isolate == 'graph':    x = x
            if self.isolate == 'features': x = additional_features

            if self.concat_normalization_flag: x = self.concat_bn(x)

        """
        if self.training:

            y_train = data.y.detach().to('cpu').tolist()
            X_train = x.detach().to('cpu').tolist()

            self.clf.fit(X_train, y_train)

        else:

            y_true = data.y.detach().to('cpu').tolist()
            X_test = x.detach().to('cpu').tolist()

            y_pred = self.clf.predict(X_test)

            calculate_metrics(y_true, y_pred, average='macro')
            print()
        """

        return x

        x = F.leaky_relu(self.linear1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(self.linear2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear3(x)

        return x

class PKGNN(torch.nn.Module):

    def __init__(self, dataset, model_params, ignore_features=False, article_feature_dict=None):

        super().__init__()

        self.feature_dict = article_feature_dict

        if self.feature_dict: model_params['model_additional_features'] = len(next(iter(article_feature_dict.values())))

        if 'model_additional_features' not in model_params: model_params['model_additional_features'] = 0

        embedding_size     = model_params["model_embedding_size"]
        n_heads            = model_params["model_attention_heads"]
        dropout_rate       = model_params["model_dropout_rate"]
        dense_neurons      = model_params["model_dense_neurons"]
        edge_dim           = model_params["model_edge_dim"]
        top_k_ratio        = model_params["model_top_k_ratio"]

        self.attention_penalty_threshold = model_params["model_attention_penalty_threshold"]

        self.top_k_every_n   = model_params["model_top_k_every_n"]
        self.ignore_features = ignore_features
        self.n_layers        = model_params["model_layers"]
        self.isolate         = model_params["model_isolate"]

        self.feature_dimensions        = model_params['model_additional_features']
        self.feature_transform_flag    = model_params['model_feature_transform_flag']
        self.concat_normalization_flag = model_params['model_concat_normalization_flag']
        self.feature_attention_flag    = model_params["model_feature_attention_flag"]
        self.top_pooling_flag          = model_params["model_top_pooling_flag"]

        self.pooling_layers = ModuleList([])
        self.conv_layers    = ModuleList([])
        self.transf_layers  = ModuleList([])
        self.bn_layers      = ModuleList([])

        self.conv1 = GATConv(
            dataset.num_node_features,
            embedding_size,
            heads    = n_heads,
            dropout  = dropout_rate,
            edge_dim = edge_dim,
            beta     = True
        )

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1     = BatchNorm1d(embedding_size)

        self.attention_penalty = 0.0

        for i in range(self.n_layers):

            self.conv_layers.append(
                TransformerConv(
                    embedding_size,
                    embedding_size,
                    heads    = n_heads,
                    dropout  = dropout_rate,
                    edge_dim = edge_dim,
                    beta     = True
                )
            )

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

            if self.top_pooling_flag and i % self.top_k_every_n == 0:

                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        if self.feature_transform_flag:

            self.feature_transform  = torch.nn.Linear(self.feature_dimensions, embedding_size)
            self.feature_dimensions = embedding_size

        if self.feature_attention_flag:

            self.attention_linear  = torch.nn.Linear(embedding_size * 2 + self.feature_dimensions, 1)
            self.attention_softmax = torch.nn.Softmax(dim=1)

        if self.isolate == 'disable':

            self.concat_bn = BatchNorm1d(embedding_size * 2 + self.feature_dimensions)
            self.linear1   = Linear(embedding_size * 2 + self.feature_dimensions, dense_neurons)

        elif self.isolate == 'features':

            self.concat_bn = BatchNorm1d(self.feature_dimensions)
            self.linear1   = Linear(self.feature_dimensions, dense_neurons)

        elif self.isolate == 'graph':

            self.concat_bn = BatchNorm1d(embedding_size * 2)
            self.linear1   = Linear(embedding_size * 2, dense_neurons)

        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))
        self.linear3 = Linear(int(dense_neurons/2), dataset.num_classes)

    """
    x          = batch.x,
    edge_index = batch.edge_index,
    target     = batch.y,
    batch      = batch.batch,
    data       = batch
    """

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.ignore_features:
            x         = torch.ones_like(x)
            edge_attr = torch.ones_like(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(self.transf1(x))

        x = self.bn1(x)

        global_representation = []

        for i in range(self.n_layers):

            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = F.leaky_relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

            if self.top_pooling_flag and (i % self.top_k_every_n == 0 or i + 1 == self.n_layers):

                x , edge_index, edge_attr, batch, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x,
                    edge_index,
                    edge_attr,
                    batch
                )

                global_representation.append(torch.cat([
                    gmp(x, batch),
                    gap(x, batch)
                ], dim=1))

            else:

                global_representation.append(torch.cat([
                    gmp(x, batch),
                    gap(x, batch)
                ], dim=1))

        ##################################
        # x = sum(global_representation) #
        ##################################

        global_representations   = torch.stack(global_representation,    dim=1)
        global_attention_weights = F.softmax(global_representations,     dim=1)
        x = torch.sum(global_attention_weights * global_representations, dim=1)

        if self.feature_dimensions > 0:

            additional_features = torch.FloatTensor([self.feature_dict[p] for p in data.path]).cuda()

            if self.feature_transform_flag:

                additional_features = self.feature_transform(additional_features)

            if self.isolate == 'disable':
                x                   = torch.cat((x, additional_features), -1)

                if self.feature_attention_flag:

                    attention_weights   = self.attention_softmax(self.attention_linear(x))
                    x                   = attention_weights * x

                    feature_weights     = attention_weights[:, -self.feature_dimensions:]
                    mean_weight         = feature_weights.mean()
                    penalty_threshold   = self.attention_penalty_threshold
                    penalty             = F.leaky_relu(mean_weight - penalty_threshold)

                    self.attention_penalty = penalty

            if self.isolate == 'graph':    x = x
            if self.isolate == 'features': x = additional_features

            if self.concat_normalization_flag: x = self.concat_bn(x)

        x = F.leaky_relu(self.linear1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(self.linear2(x))
        x = F.dropout(x, p=0.2, training=self.training)

        ###############################
        # x = F.log_softmax(x, dim=1) #
        ###############################

        x = self.linear3(x)

        return x

if __name__ == "__main__":

    micro_pkg_path = "/home/dpasch01/notebooks/PARALLAX/MicroPKGs/"
    output_dir     = "/home/dpasch01/notebooks/PARALLAX/Buzzfeed/"

    df             = PKGDataset.convert_to_dataframe(micro_pkg_path, output_dir)
    pkg_embeddings = PKGEmbeddings(output_dir)

    dataset = PKGDataset(
        root           = os.path.join(output_dir, 'parallax/dataset'),
        filename       = 'micro.csv',
        pkg_embeddings = pkg_embeddings
    )

    parameters = {
        "weight_decay":          0.00001,

        "model_attention_heads": 1,
        "batch_size": 16,
        "model_dropout_rate": 0.3,
        "scheduler_gamma": 0.9,
        "sgd_momentum": 0.8,
        "model_layers": 2,
        "learning_rate": 0.002,
        "model_embedding_size": 32,
        "model_dense_neurons": 256,

        "pos_weight" :           1,
        "model_top_k_ratio":     0.2,
        "model_top_k_every_n":   2,
    }

    train_dataset, test_dataset = dataset.split_train_test(test_ratio=0.33, batch_size=parameters['batch_size'], random_state=11)

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], collate_fn=custom_collate, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=parameters["batch_size"], collate_fn=custom_collate, shuffle=False, drop_last=True)

    parameters["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    print('Feature Size:', train_dataset[0].x.shape[1])
    print()

    # TrainingSession.hyperparameter_tuning(100, dataset)

    model_params = {k: v for k, v in parameters.items() if k.startswith("model_")}

    model = PKGNN(dataset=dataset, model_params=model_params)
    model        = model.to(DEVICE)

    loss_fn   = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = parameters["learning_rate"],
        momentum     = parameters["sgd_momentum"],
        weight_decay = parameters["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=parameters["scheduler_gamma"])

    best_loss              = 1000
    early_stopping_counter = 0

    loss_list = []

    for epoch in range(5000):

        if early_stopping_counter <= 10:

            model.train()

            loss = TrainingSession.train_epoch(epoch, model, train_loader, optimizer, loss_fn)

            classification_report(loss[1], loss[2])

            loss = loss[0]

            model.eval()

            if epoch % 5 == 0:

                train_loss = loss

                loss = TrainingSession.test(epoch, model, test_loader, loss_fn)

                classification_report(loss[1], loss[2])

                loss = loss[0]

                loss_list.append((train_loss, loss))

                if float(loss) < best_loss:
                    best_loss              = loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()

    # history_loss_model = TrainingSession.fit(
    #     parameters,
    #     dataset,
    #     train_dataset,
    #     test_dataset
    # )

    os.makedirs(os.path.join(output_dir, 'parallax/model'), exist_ok=True)

    print('Minimum Train Loss:', min([l[0] for l in loss_list]))
    print('Minimum Test  Loss:', min([l[1] for l in loss_list]))

    plt.figure(figsize=(16, 5))

    plt.plot(list(range(len(loss_list))), [t[0] for t in loss_list], label='Train')
    plt.plot(list(range(len(loss_list))), [t[1] for t in loss_list], label='Test')
    plt.legend()

    plt.savefig(os.path.join(output_dir, 'parallax/model/loss.jpg'))

    # print('Minimum Loss:', min(history_loss_model[1]))
    #
    # plt.figure(figsize=(16, 5))
    #
    # plt.plot(list(range(len(history_loss_model[0]))), [t[0] for t in history_loss_model[0]], label='Train')
    # plt.plot(list(range(len(history_loss_model[0]))), [t[1] for t in history_loss_model[0]], label='Test')
    # plt.legend()
    #
    # plt.savefig(os.path.join(output_dir, 'parallax/model/loss.jpg'))
    #
    # epoch_index = -1
    #
    # TrainingSession.log_conf_matrix(
    #     history_loss_model[0][epoch_index][2]['train_preds'],
    #     history_loss_model[0][epoch_index][2]['train_labels'],
    #     epoch_index,
    #     output = os.path.join(output_dir, 'parallax/model/conf.jpg')
    # )
    #
    # y_pred = history_loss_model[0][epoch_index][2]['test_preds']
    # y_true = history_loss_model[0][epoch_index][2]['test_labels']
    #
    # print(f"F1 Score : {f1_score(y_pred, y_true)}")
    # print(f"Accuracy : {accuracy_score(y_pred, y_true)}")
    # print(f"Precision: {precision_score(y_pred, y_true)}")
    # print(f"Recall   : {recall_score(y_pred, y_true)}")
    # print(f"ROC AUC  : {roc_auc_score(y_pred, y_true)}")
    # print()
    #
    # print(classification_report(y_pred, y_true))
    #
    # model = history_loss_model[2]

    # torch.save(model.state_dict(), os.path.join(output_dir, 'parallax/model/model.state'))
