import networkx as nx, os, numpy, pandas as pd, torch, json
from random import sample

from torch_geometric.data import Dataset, Data

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ast import literal_eval

from polarlib.parallax.generic_kg_embeddings import GenericKGEmbeddings
from polarlib.parallax.pkg_embeddings import PKGEmbeddings

class GenericKGDataset(Dataset):

    @staticmethod
    def convert_to_dataframe(micro_kg_path, output_dir, class_map=None):

        df = []

        for root, folders, files in os.walk(micro_kg_path):

            for f in files:

                f = os.path.join(root, f)

                if not f.endswith('encoded_triples.json'): continue

                with open(f, 'r') as _: triples_list = json.loads(json.load(_))['triples']

                triples_list = [tuple(t) for t in triples_list]

                if   'Unreliable' in f: label = 1
                elif 'Reliable' in f:   label = 0
                elif class_map:         label = class_map[f]

                if len(triples_list) == 0:

                    print(f'- Found empty: {label}')

                    continue

                df.append({
                    'triplets': triples_list,
                    'label':    label,
                    'path':     f
                })

        df = pd.DataFrame.from_dict(df)

        os.makedirs(os.path.join(output_dir, 'parallax/dataset/openie/raw'), exist_ok=True)

        df.to_csv(os.path.join(output_dir, 'parallax/dataset/openie/raw/micro.csv'), index=None)

        return pd.DataFrame.from_dict(df)

    def __init__(
            self,
            kg_embeddings:GenericKGEmbeddings,
            root,
            filename,
            transform         = None,
            pre_transform     = None,
            ignore_embeddings = False
    ):

        self.filename          = filename
        self.pkg_embeddings    = kg_embeddings
        self.ignore_embeddings = ignore_embeddings

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self): return self.filename
    def download(self):       pass

    def process(self):

        self.data = pd.read_csv(self.raw_paths[0])

        index = 0

        for _index, entry in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            label     = entry["label"]
            path      = entry["path"]

            entry     = literal_eval(entry['triplets'])

            entry_pkg = self.get_nx(entry)

            number_of_edges = entry_pkg.number_of_edges()

            if number_of_edges == 0:

                # print(entry)
                continue

            node_list = node_list = sorted(entry_pkg.nodes())

            node_features = self._get_node_features(entry_pkg, node_list)
            edge_features = self._get_edge_features(entry_pkg, node_list)
            edge_index    = self._get_adjacency_info(entry_pkg, node_list)

            label         = self._get_labels(label)
            if label == None: continue

            data = Data(
                x          = node_features,
                edge_index = edge_index,
                edge_attr  = edge_features,
                y          = label,
                path       = path
            )

            if data == None:
                print('Data N/A.')
                continue

            torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

            index += 1

    @property
    def processed_file_names(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        return [f'data_{i}.pt' for i in list(self.data.index)]

    def get_nx(self, entry):

        _ = nx.DiGraph()

        for e in entry:

            e = (str(e[0]), str(e[1]), str(e[2]))

            if e[0] not in self.pkg_embeddings.id_to_entity:   continue
            if e[2] not in self.pkg_embeddings.id_to_entity:   continue
            if e[1] not in self.pkg_embeddings.id_to_relation: continue

            if self.ignore_embeddings:
                _.add_node(e[0],       v=numpy.zeros(200))
                _.add_node(e[2],       v=numpy.zeros(200))
                _.add_edge(e[0], e[2], v=numpy.zeros(200))

            else:
                _.add_node(e[0], v       = self.pkg_embeddings.get_entity_vector_by_str(e[0]))
                _.add_node(e[2], v       = self.pkg_embeddings.get_entity_vector_by_str(e[2]))
                _.add_edge(e[0], e[2], v = self.pkg_embeddings.get_relation_vector_by_str(e[1]))

        return _.copy()

    def _get_node_features(self, entry_pkg, node_list):

        node_feature_list = []

        node_data = entry_pkg.nodes(data=True)

        for n in node_list: node_feature_list.append(node_data[n]['v'])

        node_feature_list = numpy.asarray(node_feature_list)

        return torch.tensor(node_feature_list, dtype=torch.float)

    def _get_edge_features(self, entry_pkg, node_list):

        edge_feature_list = []

        for n1 in node_list:

            for n2 in node_list:

                if n1 == n2: continue

                if not entry_pkg.has_edge(n1, n2): continue

                edge_feature_list.append(entry_pkg.get_edge_data(n1, n2)['v'])

        edge_feature_list = numpy.asarray(edge_feature_list)

        return torch.tensor(edge_feature_list, dtype=torch.float)

    def _get_adjacency_info(self, entry_pkg, node_list):

        edge_indices = []

        for i, n1 in enumerate(node_list):

            for j, n2 in enumerate(node_list):

                if n1 == n2: continue

                if not entry_pkg.has_edge(n1, n2): continue

                edge_indices.append([i, j])

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices

    def _get_labels(self, label):
        label = numpy.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self): return self.data.shape[0]

    def get(self, idx):

        if not os.path.exists(os.path.join(self.processed_dir, f'data_{idx}.pt')): return None

        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

        return data

    def split_train_test(self, test_ratio=0.2, random_state=11, batch_size=32):

        r_indices = [i for i,l in enumerate(self.data['label']) if l==0]
        f_indices = [i for i,l in enumerate(self.data['label']) if l==1]

        r_indices = sample(r_indices, len(f_indices))

        X_train, X_test, y_train, y_test = train_test_split(
            r_indices + f_indices,
            [0 for i in r_indices] + [1 for i in f_indices],
            test_size    = test_ratio,
            random_state = random_state,
            stratify     = [0 for i in r_indices] + [1 for i in f_indices]
        )

        train_ = (len(X_train) // batch_size) * batch_size
        test_  = (len(X_test)  // batch_size) * batch_size

        train_ = len(X_train)
        test_  = len(X_test)

        print('Train:', train_)
        print('Test: ', test_)
        print()

        train_dataset = self[:train_]
        test_dataset  = self[train_:train_ + test_]

        return train_dataset, test_dataset

if __name__ == "__main__":

    micro_pkg_path = "/home/dpasch01/notebooks/PARALLAX/MicroPKGs/"
    output_dir     = "/home/dpasch01/notebooks/PARALLAX/Buzzfeed/"

    df             = PKGDataset.convert_to_dataframe(micro_pkg_path, output_dir)
    pkg_embeddings = PKGEmbeddings(output_dir)

    dataset = PKGDataset(root=os.path.join(output_dir, 'parallax/dataset'), filename='micro.csv', pkg_embeddings=pkg_embeddings)

    train_dataset, test_dataset = dataset.split_train_test(batch_size=16)
