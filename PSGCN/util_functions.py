import os
import torch
import time
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import multiprocessing as mp

from tqdm import tqdm
from scipy import io
from torch_geometric.data import Data, InMemoryDataset, Dataset
from sklearn.model_selection import KFold

import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


class MyDataset(InMemoryDataset):
    def __init__(self, root, A, links, labels, hop):
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.hop = hop
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'data.pt'
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = links2subgraphs(self.Arow, self.Acol, self.links, self.labels, self.hop)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h

    def __len__(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, g_label, self.h)
        return construct_pyg_graph(*tmp[0:6])


def links2subgraphs(Arow, Acol, links, labels, hop):
    # extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    g_list = []
    # with tqdm(total=len(links[0])) as pbar:
    #     num = 0
    #     for i, j, g_label in zip(links[0], links[1], labels):
    #         tmp = subgraph_extraction_labeling(
    #             (i, j), Arow, Acol, g_label, hop
    #         )
    #         data = construct_pyg_graph(*tmp)
    #         g_list.append(data)
    #         pbar.update(1)
    #         num = num + tmp[-1]

    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(
        subgraph_extraction_labeling,
        [
            ((i, j), Arow, Acol, g_label)
            for i, j, g_label in zip(links[0], links[1], labels)
        ]
    )
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    end = time.time()
    print("Time elapsed for subgraph extraction: {}s".format(end - start))
    print("Transforming to pytorch_geometric graphs...")
    g_list = []
    pbar = tqdm(total=len(results))
    while results:
        tmp = results.pop()
        g_list.append(construct_pyg_graph(*tmp[0:6]))
        pbar.update(1)
    pbar.close()
    end2 = time.time()
    print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2 - end))
            
    return g_list


def subgraph_extraction_labeling(ind, Arow, Acol, label=1, h=1):

    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    for dist in range(1, h+1):
        if len(u_fringe) == 0 or len(v_fringe) == 0:
            break

        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
       
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0
    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)
    v += len(u_nodes)
    # num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
    max_node_label = 2*8 + 1  # to construct initialize label trick matrix

    return u, v, r, node_labels, max_node_label, label


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)  
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_attr=edge_type, y=y)

    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    # in case some nodes are isolated
    g.add_nodes_from(range(len(data.x)))
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g


def load_k_fold(data_name, seed):
    root_path = os.path.dirname(os.path.abspath(__file__))
    if data_name == 'lrssl':
        # txt dataset
        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.txt')
        matrix = pd.read_table(path, index_col=0).values
    elif data_name in ['Gdataset', 'Cdataset']:
        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.mat')
        # mat dataset
        data = io.loadmat(path)
        matrix = data['didr'].T
    else:
        # csv dataset
        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.csv')
        data = pd.read_csv(path, header=None)
        matrix = data.values.T

    drug_num, disease_num = matrix.shape[0], matrix.shape[1]
    drug_id, disease_id = np.nonzero(matrix)

    num_len = int(np.ceil(len(drug_id) * 1))  # setting sparse ratio
    drug_id, disease_id = drug_id[0: num_len], disease_id[0: num_len]

    neutral_flag = 0
    labels = np.full((drug_num, disease_num), neutral_flag, dtype=np.int32)
    observed_labels = [1] * len(drug_id)
    labels[drug_id, disease_id] = np.array(observed_labels)
    labels = labels.reshape([-1])

    # number of test and validation edges
    num_train = int(np.ceil(0.9 * len(drug_id)))
    num_test = int(np.ceil(0.1 * len(drug_id)))
    print("num_train {}".format(num_train),
          "num_test {}".format(num_test))

    print("num_train, num_test's ratio is", 0.9, 0.1)

    # negative sampling
    neg_drug_idx, neg_disease_idx = np.where(matrix == 0)
    neg_pairs = np.array([[dr, di] for dr, di in zip(neg_drug_idx, neg_disease_idx)])
    np.random.seed(6)
    np.random.shuffle(neg_pairs)
    # neg_pairs = neg_pairs[0:num_train + num_test - 1]
    neg_idx = np.array([dr * disease_num + di for dr, di in neg_pairs])

    # positive sampling
    pos_pairs = np.array([[dr, di] for dr, di in zip(drug_id, disease_id)])
    pos_idx = np.array([dr * disease_num + di for dr, di in pos_pairs])

    split_data_dict = {}
    count = 0
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train_data, test_data in kfold.split(pos_idx):
        # train dataset contains positive and negative
        idx_pos_train = np.array(pos_idx)[np.array(train_data)]

        idx_neg_train = neg_idx[0:len(idx_pos_train)]  # training dataset pos:neg = 1:1
        idx_train = np.concatenate([idx_pos_train, idx_neg_train], axis=0)

        pairs_pos_train = pos_pairs[np.array(train_data)]
        pairs_neg_train = neg_pairs[0:len(pairs_pos_train)]
        pairs_train = np.concatenate([pairs_pos_train, pairs_neg_train], axis=0)

        # test dataset contains positive and negative
        idx_pos_test = np.array(pos_idx)[np.array(test_data)]
        idx_neg_test = neg_idx[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test) + 1]
        idx_test = np.concatenate([idx_pos_test, idx_neg_test], axis=0)

        pairs_pos_test = pos_pairs[np.array(test_data)]
        pairs_neg_test = neg_pairs[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test) + 1]
        pairs_test = np.concatenate([pairs_pos_test, pairs_neg_test], axis=0)

        # Internally shuffle training set
        rand_idx = list(range(len(idx_train)))
        np.random.seed(42)
        np.random.shuffle(rand_idx)
        idx_train = idx_train[rand_idx]
        pairs_train = pairs_train[rand_idx]

        u_train_idx, v_train_idx = pairs_train.transpose()
        u_test_idx, v_test_idx = pairs_test.transpose()

        # create labels
        train_labels = labels[idx_train]
        test_labels = labels[idx_test]

        # make training adjacency matrix
        rating_mx_train = np.zeros(drug_num * disease_num, dtype=np.float32)
        rating_mx_train[idx_train] = labels[idx_train]
        rating_mx_train = ssp.csr_matrix(rating_mx_train.reshape(drug_num, disease_num))
        split_data_dict[count] = [rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           test_labels, u_test_idx, v_test_idx]
        count += 1

    return split_data_dict


if __name__ == '__main__':
    split_data_dict = load_k_fold('Gdataset', 1)
