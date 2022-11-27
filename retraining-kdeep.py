import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

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

def load_model(data_dict, device, args):
    gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    checkpoint = torch.load(args.trained_path, map_location=device)
    gnn.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return gnn, opt, checkpoint['epoch']

def train(data_dict, args):
    device = torch.device(args.device)
    
    # Dataset
    data_dict = prepare_dataset(data_dict, args)
    
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
        batch_size=args.batch_size,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)

    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)
    
    
    # Model
    print('Loading pretrained model...')
    gnn, opt, start_epoch = load_model(data_dict, device, args)
    start_epoch = 0


    if args.eval_epochs:
        g = data_dict['graph']
        item_ntype = data_dict['item_ntype']
        user_ntype = data_dict['user_ntype']
        user_to_item_etype = data_dict['user_to_item_etype']
        timestamp = data_dict['timestamp']
        nid_uid_dict = {v: k for v, k in enumerate(list(g.ndata['userID'].values())[0].numpy())}
        nid_wid_dict = {nid.item(): wid.item() for wid, nid in  zip(g.ndata['item_id']['item'], g.ndata['id']['item'])}

    print("------Let's start Training!------")
    best_recall = 0
    best_hitrate = 0
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        gnn.train()
        print('\033[95m' + f'----Epoch {epoch} starting!----' + '\033[0m')
        for batch in tqdm(range(args.batches_per_epoch)):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = gnn(pos_graph, neg_graph, blocks).to(device).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print status
            #if batch % 1000 == 0:
            #    print("num_epochs:", epoch, "||", "batches_per_epoch:", batch, "||", "loss:", loss.item())
        
        # Evaluate
        if not epoch:
            continue
        
        gnn.eval()
        with torch.no_grad():
            if args.eval_epochs and not epoch % args.eval_epochs:
                h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], 
                                                data_dict['textset'], item_ntype, neighbor_sampler, args.batch_size, device)
                item_batch = evaluation.item_by_user_batch(g, user_ntype, item_ntype, user_to_item_etype, timestamp, args)
                print('\033[93m' + f'----Embedding Creation Successful----' + '\033[0m')
                recalls = 0#[]
                hitrates = 0
                users = []
                num_labels = 0
                counts_n = 0

                model = NearestNeighbors(n_neighbors = args.k, 
                         metric = 'cosine',
                        )#cosine
                model.fit(h_item.detach().cpu().numpy())
                print('\033[93m' + f'----KNN fitting successful----' + '\033[0m')
                for i, nodes in tqdm(enumerate(item_batch)):
                    # 실제 유저 ID 탐색
                    category = nid_uid_dict[i]
                    user_id = data_dict['user_category'][category]  # 실제 유저 id
                    label = data_dict['validset'][user_id]  # 테스트 라벨
                    users.append(user_id)

                    # 실제 와인 ID 탐색
                    item = evaluation.node_to_item(nodes, nid_wid_dict, data_dict['item_category'])  # 와인 ID
                    label_idx = [i for i, x in enumerate(item) if x in label]  # 라벨 인덱스
                    # 아이템 추천
                    nodes = [x for i, x in enumerate(nodes)] #if i not in label_idx # 라벨 인덱스 미포함 입력 학습용 노드
                    h_nodes = h_item[nodes]
                    if h_nodes.tolist() == []:
                        continue
                    h_center = torch.mean(h_nodes, axis=0)  # 중앙 임베딩 
                    _, topk = model.kneighbors(h_center.detach().cpu().numpy().reshape(1, -1))
                    topk = topk[0]
                    tp = [x for x in label if x in topk]

                    if not tp:
                        hitrates += 0
                        counts_n += 1
                        recalls += 0
                    else:
                        recalls += len(tp)
                        num_labels += len(label)
                        hitrates += 1
                        counts_n += 1

                hitrate = hitrates / counts_n
                recall = recalls / num_labels

                print('\033[96m' + f'\tEpoch:{epoch}\tRecall:{recall}' + '\033[0m')

                if recall > best_recall:
                    if not epoch % args.save_epochs:
                        print("This is the current best score. Save the weight.")
                        torch.save({'epoch': epoch,
                                    'model_state_dict': gnn.state_dict(),
                                    'optimizer_state_dict': opt.state_dict(),
                                    'loss': loss},
                                   args.save_path + '_' + str(epoch) + 'epoch.pt')
                    best_recall = recall                      
                        
                

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default="graph_data/kdata_8.pkl")
    parser.add_argument('--trained-path', type=str, default='model/model_78epoch.pt')
    parser.add_argument('-s', '--save-path', type=str, default='model/model')
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=1024) # 128
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')  # 'cpu' or 'cuda:N'
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batches-per-epoch', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--eval-epochs', type=int, default=1)
    parser.add_argument('--save-epochs', type=int, default=1)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        
    data_dict = {
        'graph': dataset['train-graph'],
        'val_matrix': None,
        'test_matrix': None,
        'item_texts': dataset['item-texts'],
        'validset': dataset['validset'],
        'testset': dataset['testset'], 
        'user_ntype': dataset['user-type'],
        'item_ntype': dataset['item-type'],
        'user_to_item_etype': dataset['user-to-item-type'],
        'timestamp': dataset['timestamp-edge-column'],
        'user_category': dataset['user-category'], 
        'item_category': dataset['item-category']
    }
    
    # Training
    train(data_dict, args)


#     torch.save({'epoch': epoch,
#                 'model_state_dict': gnn.state_dict(),
#                 'optimizer_state_dict': opt.state_dict(),
#                 'loss': loss},
#                args.save_path + '_' + str(epoch) + 'epoch.pt')