import networkx as nx

class GenericKnowledgeGraph:

    def __init__(self, triple_list):

        self.triple_list = triple_list
        self.kg          = None

    def construct(self):

        self.kg = nx.DiGraph()

        for t in self.triple_list:

            self.kg.add_node(t[0], type='Actor')
            self.kg.add_node(t[2], type='Actor')

            self.kg.add_edge(t[0], t[2], weight=1, type='Relationship', label=t[1])
            self.kg.add_edge(t[2], t[0], weight=1, type='Relationship', label=t[1])

    def get_node_by_type(self, type='Actor'): return [kv[0] for kv in dict(self.kg.nodes(data=True)).items() if kv[1]['type'] == type]

    def _get_neighbors(self, node, attr_label=None, attr_value=None):

        u_pkg = self.kg.to_undirected(as_view=True)

        neighbors = list(u_pkg.neighbors(node))

        if attr_label == None: return neighbors

        else: return [neighbor for neighbor in neighbors if u_pkg.nodes(data=True)[neighbor].get(attr_label) == attr_value]