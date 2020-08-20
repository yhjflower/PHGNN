import networkx as nx
import numpy as np
import scipy
import pickle
import torch
import networkx as nx
from utils import *



def load_DBLP_data(prefix='../data/preprocessed/DBLP_processed'):
    # features = []
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20,dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    paper_labels=np.load(prefix + '/paper_labels.npy')
    # expected_metapaths = [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]
    #
    #
    # dblp=get_metapath_neighbor_pairs(adjM,type_mask,expected_metapaths,features_0.shape[0])
    # np.save(prefix +'metapath_idx.npy', dblp)
    # print(dblp[0])

    return features_0,features_1,features_2,features_3,adjM, type_mask,labels


def to_torch(features_0,features_1,features_2,features_3, adjM, type_mask):

    length = 10
    node_num = 0
    types =max(type_mask)+1

    edge = adjM.nonzero()
    edge_index=[]
    for i in range(len(edge[0])):
        edge_index.append([edge[0][i],edge[1][i]])
    graph = nx.Graph()
    graph.add_edges_from(edge_index)
    h_mat = hete_mat(graph, types, type_mask)



    # node2vec2 = Node2Vec(graph, walk_length=length, num_walks=1, workers=1)
    # walks2 = node2vec2.walks
    # edge_index_2 = random_walk(walks2, h_mat, edge_index,2)
    # np.save('../data/preprocessed/DBLP_processed/' + 'edge_index.npy',edge_index_2)
    # edge_index_2=np.load('../data/preprocessed/IMDB_processed/' + 'edge_index.npy')

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    node_type = torch.LongTensor(type_mask)
    h_mat = standardization(h_mat)
    edge_index = torch.LongTensor(edge_index).t()
    h_mat = torch.FloatTensor(h_mat)
    print(edge_index.shape)


    return features_0,features_1,features_2,features_3, node_type,edge_index,h_mat

def random_walk(walks,h_mat,edge_index1,k):

    edge_index=edge_index1.copy()
    for i in range(len(walks)):
        dif=(h_mat[walks[i,1:]]-h_mat[walks[i,0]]).sum(1)
        ind=np.argpartition(dif, k)[:k]
        for t in range(k):
            edge_index.append([walks[i,0],walks[i,ind[t]]])
    return np.unique(edge_index,axis=0)


def hete_mat(graph,types,node_type):
    het_matrix_1=[]
    het_matrix_2=[]
    for n in graph.nodes():
        h_1 = np.zeros(types)
        for neigh in graph.neighbors(n):
            h_1[node_type[neigh]]+=1
        het_matrix_1.append(h_1)
    het_matrix_1=np.asarray(het_matrix_1)
    for n in graph.nodes():
        neigh=list(graph.neighbors(n))
        chosed_neigh=np.random.choice(neigh,4)
        h_2=np.mean(het_matrix_1[chosed_neigh],axis=0)
        het_matrix_2.append(h_2)
    het_matrix=np.concatenate((het_matrix_1,het_matrix_2),1)
    return np.array(het_matrix)

def standardization(x):
    x=x.T
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x= (x - mu) / (sigma+1e-3)
    return x.T