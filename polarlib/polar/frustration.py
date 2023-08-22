import networkx as nx, numpy as np, time, itertools

def G_to_fi(G):

    """
    Convert a signed network graph to frustration index related data.

    Parameters:
    - G (networkx.Graph): Input graph.

    Returns:
    - sign_G (networkx.Graph): A graph with nodes and edges representing positive and negative edges.
    - adj_sign_G (numpy.matrix): Adjacency matrix of the sign_G graph.
    - sign_edgelist (dict): Dictionary of edge weights for positive and negative edges.
    - id_to_node (dict): Mapping of node indices to their original names.
    """

    nodelist = G.nodes()
    edgelist = G.edges(data=True)
    
    node_to_id = {}
    for i in range(len(nodelist)):
        n = list(nodelist)[i]
        node_to_id[n] = i
        
    id_to_node = {i:n for n,i in node_to_id.items()}
    
    pos_edgelist = [(node_to_id[e[0]], node_to_id[e[1]], {'weight': 1}) for e in list(edgelist) if e[2]['weight'] > 0.0]
    neg_edgelist = [(node_to_id[e[0]], node_to_id[e[1]], {'weight': -1}) for e in list(edgelist) if e[2]['weight'] < 0.0]
    
    sign_G = nx.Graph()
    sign_G.add_nodes_from(list(node_to_id.values()))
    sign_G.add_edges_from(pos_edgelist + neg_edgelist)
    
    adj_sign_G = nx.adjacency_matrix(sign_G, nodelist=list(node_to_id.values())).todense()
    adj_unsign_G = abs(nx.adjacency_matrix(sign_G, nodelist=list(node_to_id.values()))).todense()
    
    sign_edgelist = nx.get_edge_attributes(sign_G, 'weight')
    
    return sign_G, adj_sign_G, sign_edgelist, id_to_node

from gurobipy import *
import multiprocessing

def calculate_frustration_index(sign_G, sign_matrix, sign_edgelist):
    """
    Calculate the frustration index of a signed network graph.

    Parameters:
    - sign_G (networkx.Graph): Graph with positive and negative edges.
    - sign_matrix (numpy.matrix): Matrix representing edge signs.
    - sign_edgelist (dict): Dictionary of edge weights for positive and negative edges.

    Returns:
    - f_g (float): Frustration index value.
    - rounded_obj_value (float): Rounded objective value.
    - solve_time (float): Time taken to solve the optimization model.
    - optimal_solution (dict): Optimal variable values.
    """

    order = len(sign_matrix)
    number_of_negative = ((-1 == sign_matrix)).sum()/2
    size = int(np.count_nonzero(sign_matrix)/2)

    neighbors, degree = {}, []
    for u in sorted(sign_G.nodes()):
        neighbors[u] = list(sign_G[u])
        degree.append(len(neighbors[u]))
        
    unsigned_degree = degree
    
    maximum_degree = max(unsigned_degree)
    [node_to_fix] = [([i for i, j in enumerate(unsigned_degree) if j == maximum_degree]).pop()]
    
    model = Model("Continuous model for lower-bounding frustration index")
    model.setParam(GRB.param.OutputFlag, 0) 
    model.setParam(GRB.param.Method, 2)
    model.setParam(GRB.Param.Crossover, 0)
    model.setParam('TimeLimit', 10 * 3600)
    model.setParam(GRB.Param.Threads, min(32, multiprocessing.cpu_count()))
   
    graph_triangles = []
    for n1 in sorted(sign_G.nodes()):
        neighbors1 = set(sign_G[n1])
        
        for n2 in filter(lambda x: x > n1, neighbors1):
            neighbors2 = set(sign_G[n2])
            common = neighbors1 & neighbors2
            for n3 in filter(lambda x: x > n2, common): graph_triangles.append([n1 ,n2, n3])
    
    z, x = {}, []
    for i in range(0, order): x.append(model.addVar(lb=0.0, ub=1, vtype=GRB.CONTINUOUS, name='x'+str(i))) 
    for (i, j) in (sign_edgelist): z[(i,j)] = model.addVar(
        lb = 0.0,
        ub = 1,
        vtype = GRB.CONTINUOUS,
        name = 'z' + str(i) + ',' + str(j)
    )    

    model.update()
    
    OFV=0
    for (i,j) in (sign_edgelist):
        OFV += (1-(sign_edgelist)[(i,j)])/2 + ((sign_edgelist)[(i,j)]) * (x[i] + x[j] -2 * z[(i,j)]) 
        
    model.setObjective(OFV, GRB.MINIMIZE)

    for (i, j) in (sign_edgelist):
            if sign_edgelist[(i, j)]==1:
                model.addConstr(z[(i, j)] <= (x[i] + x[j]) / 2 , 'Edge positive' + str(i) + ',' + str(j))
            if (sign_edgelist)[(i, j)]==-1:
                model.addConstr(z[(i, j)] >= x[i] + x[j] -1 , 'Edge negative' + str(i) + ',' + str(j))            

    for triangle in graph_triangles:
            [i, j, k] = triangle
            
            model.addConstr(x[j] + z[(i, k)] >= z[(i, j)] + z[(j, k)] , 'triangle1' + ',' + str(i) + ',' + str(j) + ',' + str(k))
            model.addConstr(x[i] + z[(j, k)] >= z[(i, j)] + z[(i, k)] , 'triangle2' + ',' + str(i) + ',' + str(j) + ',' + str(k))       
            model.addConstr(x[k] + z[(i, j)] >= z[(i, k)] + z[(j, k)] , 'triangle3' + ',' + str(i) + ',' + str(j) + ',' + str(k))
            model.addConstr( 1 + z[(i, j)] + z[(i, k)] + z[(j, k)] >= x[i] + x[j] + x[k], 'triangle4' + ',' + str(i) + ',' + str(j) + ',' + str(k))           
            
    model.update()
    model.addConstr(x[node_to_fix]==1 , '1stnodecolour')   
    model.update()
  
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    obj = model.getObjective()
    
    optimal_solution = {'x':{}, 'z':{}}
    
    for v in model.getVars(): 
        if 'x' in v.varName: optimal_solution['x'][v.varName[1:]] = v.x
        elif 'z' in v.varName: optimal_solution['z'][v.varName[1:]] = v.x
    
    f_g = 1.0 - 2 * np.around(obj.getValue()) / len(sign_edgelist) if len(sign_edgelist) > 0 else 0.0
    
    return f_g, np.around(obj.getValue()), solve_time, optimal_solution

def triadic_balance(G, triad):
    """
    Calculate the triadic balance of a triad in a signed network graph.

    Parameters:
    - G (networkx.Graph): Graph containing the triad.
    - triad (tuple): Tuple containing nodes of the triad.

    Returns:
    - balance_str (str): String representing the triadic balance (+ and - signs).
    """
    plus_str, minus_str = "", ""
    for c in itertools.combinations(triad, 2):
        if G.get_edge_data(c[0], c[1])['weight'] < 0.0: minus_str += '-'
        else: plus_str += '+'
            
    return plus_str + minus_str

def calc_triadic_balance(G):
    """
    Calculate the distribution of triadic balances in a signed network graph.

    Parameters:
    - G (networkx.Graph): Input graph.

    Returns:
    - gtb_dict (dict): Dictionary containing the distribution of triadic balances.
    """
    gtb_dict = {
        '+++': 0,
        '+--': 0,
        '++-': 0,
        '---': 0,
    }
    
    G_cliques = nx.enumerate_all_cliques(G)
    G_triads = [x for x in G_cliques if len(x)==3]
    
    if len(G_triads) == 0: return gtb_dict
    
    for triad in G_triads:
        tb = triadic_balance(G, triad)
        gtb_dict[tb] += 1
        
    for key in gtb_dict:
        gtb_dict[key] /= len(G_triads)
        
    return gtb_dict
