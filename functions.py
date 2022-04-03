import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import itertools
import random
from datetime import timedelta


def create_graph_from_time(data, column1, column2, t_min, t_max, weight=None, directed=False):
    # берём конкретный период
    data = data.loc[data.DateTime.between(t_min, t_max)]

    # за вес берём количество интеракций
    if weight is None:
        data = data.groupby([column1, column2], as_index=False).size()
        data = data.rename(columns={'size': 'weight'})
    # каждая интеракция берется с её весом в виде длительности и все суммируется
    else:
        data = data.groupby([column1, column2], as_index=False)[weight].sum()
        data = data.rename(columns={weight: 'weight'})

    # ненаправленный граф
    if not directed:
        graph = nx.from_pandas_edgelist(data, column1, column2, 'weight')

    # направленный граф
    else:
        graph = nx.from_pandas_edgelist(data, column1, column2, edge_attr='weight', create_using=nx.DiGraph())

    return graph


def create_graph_from_time_multi_edges(data, column1, column2, t_min, t_max):
    # берём конкретный период
    data = data.loc[data.DateTime.between(t_min, t_max)]

    graph = nx.from_pandas_edgelist(data, column1, column2, create_using=nx.MultiGraph())

    return graph


def draw_graph(G, pos=None, factor=500):
    if pos is None:
        pos = nx.spring_layout(G, weight='weight')

    plt.figure(figsize=(15, 15))
    # толщина ребра будет зависить от веса, factor - для адекватности размера
    weights = np.array(list(nx.get_edge_attributes(G, 'weight').values())) / factor

    # размер вершины от степени
    d = nx.degree(G)
    d = [(d[node]+1) * 200 for node in G.nodes()]

    if type(G) is not nx.classes.digraph.DiGraph:
        nx.draw(G, pos, width=weights, node_size=d, with_labels=True)

    # доп. параметры для нормальной отрисовки направленного графа
    else:
        nx.draw(G, pos, width=weights, node_size=d, with_labels=True, connectionstyle='arc3, rad = 1')
        
        
def prepare_OBS_and_RFID():
    OBS = pd.read_csv('OBS_data.txt', sep='\t')
    OBS.DateTime = pd.to_datetime(OBS.DateTime, infer_datetime_format=True)
    OBS = OBS.dropna()
    OBS = OBS[OBS.Recipient != 'SELF']
    OBS.Recipient = OBS.Recipient.apply(lambda x: x.strip())
    
    le = LabelEncoder()
    le.fit(np.unique(OBS[['Actor', 'Recipient']].values))
    
    OBS.Actor = le.transform(OBS.Actor)
    OBS.Recipient = le.transform(OBS.Recipient)
    OBS_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    OBS['Day'] = OBS.DateTime.apply(lambda x: x.day)
    
    RFID = pd.read_csv('RFID_data.txt', sep='\t')
    RFID.DateTime = pd.to_datetime(RFID.DateTime, infer_datetime_format=True)
    RFID.i = le.transform(RFID.i)
    RFID.j = le.transform(RFID.j)
    return OBS, RFID, OBS_name_mapping


def count_motifs(gr):
    
    motifs = {
    'S1': nx.DiGraph([(1,2),(2,3)]),
    'S2': nx.DiGraph([(1,2),(1,3),(2,3)]),
    'S3': nx.DiGraph([(1,2),(2,3),(3,1)]),
    'S4': nx.DiGraph([(1,2),(3,2)]),
    'S5': nx.DiGraph([(1,2),(1,3)])
    }

    nodes = gr.nodes()
    
    mcount = defaultdict(int)

    triplets = list(itertools.product(*[nodes, nodes, nodes]))
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]


    for trip in u_triplets:
        sub_gr = gr.subgraph(trip)
        match_keys = []
        for i in motofs.keys():
            if nx.is_isomorphic(sub_gr, motifs[i]):
                mcount[i] += 1

    return mcount, motifs


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def create_matrix(days, names, res):
    pairs = list(combinations(names, 2))
    matrix = np.zeros((len(days), len(pairs)), dtype=int)
    for idx1, day in enumerate(days):
        for idx2, pair in enumerate(pairs):
            matrix[idx1, idx2] = res[day].count(pair)
    
    answer = np.zeros((len(days), len(days)))
    for i in range(len(days)):
        for j in range(len(days)):
            answer[i, j] = cos_sim(matrix[i, :], matrix[j, :])
    
    return answer   

    
def create_similarity_matrix(df, names, days):
    cosine = np.zeros((len(days), len(days)))
    day_edges = defaultdict(list)
    for idx, day in enumerate(days):
        day_df = df[df['Day'] == day]
        for _, row in day_df.iterrows():
            day_edges[day].append(tuple(sorted((row['Actor'], row['Recipient']))))
        
    return create_matrix(days, names, day_edges)
 
 
community_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
    3 : 'tab:red',
    4 : 'tab:purple',
    5 : 'tab:pink',
    6 : 'tab:red',
    7 : 'tab:gray'
}
   
    
def draw_community_graph(G, pos=None, factor=500, colors = None):
    if pos is None:
        pos=nx.spring_layout(G)
    edges = G.edges() # толщина ребра будет зависить от веса
    weights = np.array(list(nx.get_edge_attributes(G,'weight').values()))/factor #чтобы размер был адекватным
    d = nx.degree(G)
    d = [(d[node]+1) * 50 for node in G.nodes()] # размер вершины от степени
    if type(G) is not  nx.classes.digraph.DiGraph:
        nx.draw(G, pos, width=weights, node_size = d, with_labels=True, node_color=colors)
    else: # доп. параметры для нормальной отрисовки направленного графа
        nx.draw(G, pos, width=weights, node_size = d, with_labels=True, connectionstyle='arc3, rad = 1',
               node_color=colors)
        
def get_colors(G, communities):
    colors = dict()
    for idx, community in enumerate(communities):
        for node in community:
            colors[node] = community_to_color[idx]
    colors = [colors[i] for i in G.nodes]
    return colors
    
    
def draw_dynamic_communities(data, graph_params, algorithm, alghorithm_params, step=5, pos=None, size=3, factor=100):
    
    start = data.DateTime.min()
    step = timedelta(days=step)
    finish = start + step
    max_date = data.DateTime.max()
    i = 1

    while start < max_date:
        G = create_graph_from_time(data, **graph_params, t_min=start, t_max=finish)
        if start == data.DateTime.min() and pos is None:
            pos = nx.spring_layout(G)
        communs = algorithm(G, **alghorithm_params)
        cols = get_colors(G, communs)
        plt.subplot(size,size,i)
        draw_community_graph(G, pos = pos, colors = cols, factor=factor) 
        start = start + step
        finish = start + step  if finish + step < max_date else max_date
        i += 1
    plt.show()
    
    
def create_random_multigraph(n, m, all_edges=None):
    if all_edges is None:
        all_edges = list(combinations(range(n), 2))
    edges = random.choices(all_edges, k=m)
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def calculate_func(func, dt_min, dt_max, df=None, nodes_cnt=20, sample_size=20, n=100):
    coefs = []
    if df is not None:
        big_G = create_graph_from_time_multi_edges(df, 'Actor', 'Recipient', dt_min, dt_max)      
        edges = [e for e in big_G.edges]
    else:
        all_edges = list(combinations(range(nodes_cnt), 2))
    for _ in range(n):
        if df is None:
            G = create_random_multigraph(nodes_cnt, sample_size, all_edges)
        else:
            edge_list = random.sample(edges, sample_size)
            G = big_G.edge_subgraph(edge_list).copy()
            G.add_nodes_from(range(nodes_cnt))
        coefs.append(func(G))
    return coefs



