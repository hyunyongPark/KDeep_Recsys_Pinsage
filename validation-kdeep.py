import os
import sys
import platform ,psutil

import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
#import torchtext
import torchtext.legacy as torchtext
from torch.utils.data import DataLoader

import layers
import sampler as sampler_module
import evaluation
from ranger import Ranger

from scipy import spatial
from sklearn.neighbors import NearestNeighbors

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)
        
def load_model(data_dict, args):
    gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers).to(args.device)
    #opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    checkpoint = torch.load(args.model_path + '.pt', map_location="cuda:0")
    gnn.load_state_dict(checkpoint['model_state_dict'])
    
    return gnn

def prepare_dataset(data_dict, args):
    g = data_dict['graph']
    item_texts = data_dict['item_texts']
    user_ntype = data_dict['user_ntype']
    item_ntype = data_dict['item_ntype']

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))
    data_dict['graph'] = g

    # Prepare torchtext dataset and vocabulary
    if not len(item_texts):
        data_dict['textset'] = None
    else:
        fields = {}
        examples = []
        for key, texts in item_texts.items():
            fields[key] = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
        for i in range(g.number_of_nodes(item_ntype)):
            example = torchtext.data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()])
            examples.append(example)
            
        textset = torchtext.data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
            #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')
        data_dict['textset'] = textset

    return data_dict

def prepare_dataloader(data_dict, args):
    g = data_dict['graph']
    user_ntype = data_dict['user_ntype']
    item_ntype = data_dict['item_ntype']
    textset = data_dict['textset']
    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)

    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)

    return dataloader_it, dataloader_test, neighbor_sampler

def valid(data_dict, args):
    device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Current using CPUs')
    else:
        print ('Current cuda device ', torch.cuda.current_device()) # check

    # Dataset
    data_dict = prepare_dataset(data_dict, args)
    dataloader_it, dataloader_test, neighbor_sampler = prepare_dataloader(data_dict, args)

    gnn = load_model(data_dict, args)

    g = data_dict['graph']
    item_ntype = data_dict['item_ntype']
    user_ntype = data_dict['user_ntype']
    user_to_item_etype = data_dict['user_to_item_etype']
    timestamp = data_dict['timestamp']
    nid_uid_dict = {v: k for v, k in enumerate(list(g.ndata['userID'].values())[0].numpy())}
    nid_wid_dict = {nid.item(): wid.item() for wid, nid in zip(g.ndata['item_id']['item'], g.ndata['id']['item'])}

    gnn = gnn.to(device)
    print('\033[95m' + f'----Creating embeddings for all items----' + '\033[0m')
    h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], 
                                    data_dict['textset'], item_ntype, neighbor_sampler, args.batch_size, device)
    item_batch = evaluation.item_by_user_batch(g, user_ntype, item_ntype, user_to_item_etype, timestamp, args)
    
    hitrates = 0
    users = []
    counts_n = 0
    model = NearestNeighbors(n_neighbors = args.k, 
                             metric = 'cosine',
                            )#cosine
    model.fit(h_item.detach().cpu().numpy())
    print('\033[94m' + f'----Success----' + '\033[0m')
    print('\033[95m' + f'----Measuring performance for testset----' + '\033[0m')
    for i, nodes in tqdm(enumerate(item_batch)):
        # 실제 유저 ID 탐색
        category = nid_uid_dict[i]
        user_id = data_dict['user_category'][category]
        label = data_dict['testset'][user_id]
        users.append(user_id)

        item = evaluation.node_to_item(nodes, nid_wid_dict, data_dict['item_category'])
        label_idx = [i for i, x in enumerate(item) if x in label]
        nodes = [x for i, x in enumerate(nodes)if i not in label_idx]
        h_nodes = h_item[nodes]
        h_center = torch.mean(h_nodes, axis=0) 
        _, topk = model.kneighbors(h_center.detach().cpu().numpy().reshape(1, -1))

        topk = topk[0]
        label = [list(label)[0]]
        tp = [x for x in label if x in topk]
        if not tp:
            hitrates += 0
            counts_n += 1
        else:
            hitrates += 1  # 하나라도 있음
            counts_n += 1

    hitrate = hitrates / counts_n
    print('\033[96m' + f'\tHR@{args.k}:{hitrate}' + '\033[0m')
                

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default="graph_data/kdata_entire8.pkl")
    parser.add_argument('-s', '--model-path', type=str, default="model/model_best")
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=1024) # 128
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')  # 'cpu' or 'cuda:N'
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('-k', type=int, default=500)
    args = parser.parse_args()
    
    print('\033[94m' + f'----Training Environment----' + '\033[0m')
    info = platform.uname()
    print('OS                   :\t', platform.system())
    print('OS Version           :\t', platform.version())
    print('Process information  :\t', platform.processor())
    print('Process Architecture :\t', platform.machine())
    print('RAM Size             :\t',str(round(psutil.virtual_memory().total / (1024.0 **3)))+"(GB)")
    print(f"CPU : {info.processor}")
    print(f"GPU : {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"python version {sys.version}")
    print(f"torch version {torch.__version__}")
    
    print();print()
    
    print('\033[94m' + f'----Training parameters----' + '\033[0m')
    print(f"random walk length : {args.random_walk_length}")
    print(f"random walk restart probability : {args.random_walk_restart_prob}")
    print(f"number of random-walks : {args.num_random_walks}")
    print(f"number of neighbors (sampling) : {args.random_walk_length}")
    print(f"number of layers : {args.num_layers}")
    print(f"hidden layers dimension : {args.hidden_dims}")
    print(f"batch size : {args.batch_size}")
    print(f"number of workers : {args.num_workers}")
    print(f"Number of K for verification : {args.k}")
    print("optimizer : Adam")
    print("Epochs : 200")
    print("batches per epoch : 10000")
    
    print();print()
    
    print('\033[94m' + f'----Ratio of datasets----' + '\033[0m')
    print(f"Train Dataset : 199,923")
    print(f"Validation Dataset : 24,929")
    print(f"Test Dataset : 25,645")
    
    print();print()
    
    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        
    data_dict = {
            'graph': dataset['train-graph'],
            'val_matrix': None,
            'test_matrix': None,
            'item_texts': dataset['item-texts'],
            'testset': dataset['testset'], 
            'user_ntype': dataset['user-type'],
            'item_ntype': dataset['item-type'],
            'user_to_item_etype': dataset['user-to-item-type'],
            'timestamp': dataset['timestamp-edge-column'],
            'user_category': dataset['user-category'], 
            'item_category': dataset['item-category']
        }
    
    valid(data_dict, args)