import pandas as pd, os, pykeen, numpy

from tabulate import tabulate
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from polarlib.prism.polarization_knowledge_graph import PolarizationKnowledgeGraph

print()
pykeen.env()
print()

import os, pandas as pd, torch

class PKGEmbeddings:

    def __init__(self, output_dir, embedding_dir=None, model_str='TuckER'):

        self.output_dir   = output_dir
        self.parallax_dir = os.path.join(output_dir, 'parallax')
        self.model_str    = model_str

        if embedding_dir == None:
            self.entity_path    = os.path.join(self.parallax_dir, 'global/{}/training_triples/entity_to_id.tsv.gz'.format(self.model_str.lower()))
            self.relation_path  = os.path.join(self.parallax_dir, 'global/{}/training_triples/relation_to_id.tsv.gz'.format(self.model_str.lower()))
            self.emb_model_path = os.path.join(self.parallax_dir, 'global/{}/trained_model.pkl'.format(self.model_str.lower()))
        else:
            self.entity_path = os.path.join(self.parallax_dir,    '{}/{}/training_triples/entity_to_id.tsv.gz'.format(embedding_dir, self.model_str.lower()))
            self.relation_path = os.path.join(self.parallax_dir,  '{}/{}/training_triples/relation_to_id.tsv.gz'.format(embedding_dir, self.model_str.lower()))
            self.emb_model_path = os.path.join(self.parallax_dir, '{}/{}/trained_model.pkl'.format(embedding_dir, self.model_str.lower()))

        self.entity_to_id   = pd.read_csv(self.entity_path,   compression='gzip', header=0, sep='\t', quotechar='"')
        self.relation_to_id = pd.read_csv(self.relation_path, compression='gzip', header=0, sep='\t', quotechar='"')

        self.entity_to_id   = self.entity_to_id.set_index('id')
        self.entity_to_id   = self.entity_to_id.T.to_dict()

        self.relation_to_id = self.relation_to_id.set_index('id')
        self.relation_to_id = self.relation_to_id.T.to_dict()

        self.entity_to_id   = {k:v['label'] for k,v in self.entity_to_id.items()}
        self.relation_to_id = {k:v['label'] for k,v in self.relation_to_id.items()}

        self.id_to_entity   = {v:k for k,v in self.entity_to_id.items()}
        self.id_to_relation = {v:k for k,v in self.relation_to_id.items()}

        for r in self.id_to_relation: print('-', r)

        self.emb_model = torch.load(self.emb_model_path, map_location=torch.device('cpu'))

        self.entity_embeddings   = self.emb_model.entity_representations[0](indices=None).detach().cpu().numpy()
        self.relation_embeddings = self.emb_model.relation_representations[0](indices=None).detach().cpu().numpy()

    def get_entity_id(self, entity):     return self.id_to_entity[entity]
    def get_relation_id(self, relation): return self.id_to_relation[relation]

    def get_entity_vector_by_str(self, entity):       return self.entity_embeddings[self.get_entity_id(entity)]
    def get_entity_vector_by_id(self, entity_id):     return self.entity_embeddings[entity_id]
    def get_relation_vector_by_str(self, relation):   return self.relation_embeddings[self.get_relation_id(relation)]
    def get_relation_vector_by_id(self, relation_id): return self.relation_embeddings[relation_id]

class PKGEmbedder:

    def __init__(self, pkg:PolarizationKnowledgeGraph, output_dir):

        self.pkg          = pkg
        self.output_dir   = output_dir
        self.parallax_dir = os.path.join(output_dir, 'parallax')

        os.makedirs(self.parallax_dir, exist_ok=True)

    def construct_embeddings(self, model_str='TuckER', output_dir=None, schema_function=None, evaluator_str='RankBasedEvaluator', epochs=5, training_loop_str='sLCWA'):

        pkg_triplet_list = self.get_pkg_triples(schema_function=schema_function)
        pkg_triplet_list = [(t['subject'], t['predicate'], t['object']) for t in pkg_triplet_list.T.to_dict().values()]

        tf = TriplesFactory.from_labeled_triples(numpy.asarray(pkg_triplet_list))

        pkg_train, pkg_test = tf.split()

        result = pipeline(
            model         = model_str,
            training      = pkg_train,
            testing       = pkg_test,
            epochs        = epochs,
            evaluator     = evaluator_str,
            training_loop = training_loop_str,

            # model_kwargs         = {'automatic_memory_optimization': True},
            # training_kwargs      = {'automatic_memory_optimization': True},

            training_loop_kwargs = {'automatic_memory_optimization': True},
            evaluator_kwargs     = {'batch_size': 256}
        )

        if output_dir == None: result.save_to_directory(os.path.join(self.parallax_dir, 'global/{}'.format(model_str.lower())))
        else: result.save_to_directory(os.path.join(self.parallax_dir, '{}/{}'.format(output_dir, model_str.lower())))

        df = result.metric_results.to_df()

        print(df.columns)

        # df = df[df['Type'] == 'realistic']
        df = df[df['Side'] == 'both']

        print(tabulate(df[df['Metric'] == 'adjusted_arithmetic_mean_rank'], headers = 'keys', tablefmt = 'psql'))

    def get_pkg_triples(self, schema_function=None, output=None):

        triples_list = []

        for e in self.pkg.pkg.edges(data=True):

            triples_list.append((e[0], e[2]['predicate'].upper() if 'predicate' in e[2] else e[2]['type'].upper(), e[1]))

        if schema_function:

            triples_list = schema_function(triples_list)

        triples_list = [
            {
                'subject':   t[0],
                'predicate': t[1],
                'object':    t[2]
            }
            for t in triples_list if not t[1] == 'NEUTRAL'
        ]

        df = pd.DataFrame.from_dict(triples_list)

        if not output==None:

            df.to_csv(os.path.join(self.parallax_dir, 'triples.csv'), index=None)

        return df

if __name__ == "__main__":

    output_dir = "/home/dpasch01/notebooks/PARALLAX/Buzzfeed/"

    pkg = PolarizationKnowledgeGraph(output_dir)

    pkg.construct()

    pkg_embedder = PKGEmbedder(pkg, output_dir)
    df           = pkg_embedder.get_pkg_triples()

    pkg_embedder.construct_embeddings()

    pkg_embs = PKGEmbeddings(output_dir)

    print(pkg_embs.get_entity_id('F1'))
    print(pkg_embs.get_relation_id('MEMBER'))

    print()

    print(pkg_embs.get_entity_vector_by_str('F1'))
    print(pkg_embs.get_relation_vector_by_str('MEMBER'))

    print()

    print(pkg_embs.get_entity_vector_by_str('F1').all() == pkg_embs.get_entity_vector_by_id(542).all())
    print(pkg_embs.get_relation_vector_by_str('MEMBER').all() == pkg_embs.get_relation_vector_by_id(1).all())
