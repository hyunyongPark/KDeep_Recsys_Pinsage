import os
import re
import random
import json
import pickle
import argparse
import ast
import scipy.sparse as ssp
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np
import dgl
from builder import PandasGraphBuilder
from data_utils import *


def build_graph(args):
    users = pd.read_csv(os.path.join(args.directory, "user_data.csv"), index_col=0)
    users.head(2)

    columns = ['user_name', 'r_gender', 'age']
    #['user_name', 'r_gender', 'age', "mar", "job", "income", "r_style1", "r_style2", "r_style3", "r_style4", "r_style5"]
    users = users[columns]
    users.columns = ['userID', 'r_gender', 'age']
    users = users.dropna(subset=['userID'])

    users = pd.get_dummies(users, columns = ['r_gender'])
    users['user_feats'] = list(users[['r_gender_1', 'r_gender_2']].values)
    del users["age"]

    items = pd.read_csv(os.path.join(args.directory, "item_data.csv"), index_col=0)
    items.head(2)

    columns = ['item', 'era', 'style', 'gender', 'season'] + \
    ['tpo','fit','brightness','temperature','weight','nice_nice','nice_no','urban_no','urban_urban',
     'trendy_no','trendy_trendy','sophisticated_no','sophisticated_sophisticated','clean_clean','clean_no',
     'magnificent_magnificent','magnificent_no','unique_no','unique_unique','easy_easy','easy_no',
     'open_no','open_open mined','practical_no','practical_practical','activity_activity','activity_no',
     'comfortable_comfortable','comfortable_no','bubbly_bubbly','bubbly_no',
     'feminine_feminine','feminine_no','manly_manly','manly_no','soft_no','soft_soft']

    items = items[columns]

    items.columns = ['item_id', 'era', 'style', 'gender', 'season'] + \
    ['tpo','fit','brightness','temperature','weight','nice_nice','nice_no','urban_no','urban_urban',
     'trendy_no','trendy_trendy','sophisticated_no','sophisticated_sophisticated','clean_clean','clean_no',
     'magnificent_magnificent','magnificent_no','unique_no','unique_unique','easy_easy','easy_no',
     'open_no','open_open mined','practical_no','practical_practical','activity_activity','activity_no',
     'comfortable_comfortable','comfortable_no','bubbly_bubbly','bubbly_no',
     'feminine_feminine','feminine_no','manly_manly','manly_no','soft_no','soft_soft']
    items = items.dropna(subset=['item_id'])
    items = pd.get_dummies(items, columns = ["style", "era", 'gender'])
    cat_columns = items.columns.drop(["item_id", "season", "tpo", "fit", "brightness", "temperature", "weight"])
    items['item_feats'] = list(items[cat_columns].values)
    items = items[["item_id", "season", "tpo", "fit", "brightness", "temperature", "weight", "item_feats"]]
    ratings = pd.read_csv(os.path.join(args.directory, "rate_data.csv"), index_col=0)

    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = ratings['user'].unique()
    distinct_items_in_ratings = ratings['item'].unique()
    users = users[users['userID'].isin(distinct_users_in_ratings)]
    items = items[items['item_id'].isin(distinct_items_in_ratings)]

    ratings.columns = ["userID", "item_id", "rating_per_user"]

    # Build Graph
    # 아이템, 유저 DB에 존재하는 rating만 사용
    user_intersect = set(ratings['userID'].values) & set(users['userID'].values)
    item_intersect = set(ratings['item_id'].values) & set(items['item_id'].values)

    new_users = users[users['userID'].isin(user_intersect)]
    new_items = items[items['item_id'].isin(item_intersect)]
    new_ratings = ratings[ratings['userID'].isin(user_intersect) & ratings['item_id'].isin(item_intersect)]
    new_ratings = new_ratings.sort_values('userID')

    label = []
    for userID, df in new_ratings.groupby('userID'):
        idx = int(df.shape[0] * 0.8)
        idx2 = int(df.shape[0] * 0.9)
        timestamp = [0] * df.shape[0]
        tstamp = []
        for i, x in enumerate(timestamp):
            if idx <= i < idx2:
                tstamp.append(1)
            elif i >= idx2:
                tstamp.append(2)
            else:
                tstamp.append(x)
        label.extend(tstamp)
        #break
    new_ratings['timestamp'] = label
    print(new_ratings.loc[new_ratings['timestamp'] == 0, :].shape)
    print(new_ratings.loc[new_ratings['timestamp'] == 1, :].shape)
    print(new_ratings.loc[new_ratings['timestamp'] == 2, :].shape)



    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'userID', 'user')
    graph_builder.add_entities(items, 'item_id', 'item')
    graph_builder.add_binary_relations(new_ratings, 'userID', 'item_id', 'rated')
    graph_builder.add_binary_relations(new_ratings, 'item_id', 'userID', 'rated-by')
    g = graph_builder.build()
    # Assign features.
    node_dict = { 
        'user': [users, ['userID', 'user_feats'], ['cat', 'int']],
        'item': [items, ['item_id', 'item_feats'], ['cat', 'int']]
    }
    edge_dict = { 
        'rated': [new_ratings, ['rating_per_user', 'timestamp']],
        'rated-by': [new_ratings, ['rating_per_user', 'timestamp']]
    }
    for key, (df, features ,dtypes) in node_dict.items():
        for value, dtype in zip(features, dtypes):
            # key = 'user' or 'wine'
            # value = 'user_follower_count' 등등
            if dtype == 'int':
                array = np.array([i for i in df[value].values])
                g.nodes[key].data[value] = torch.FloatTensor(array)
            elif dtype == 'cat':
                g.nodes[key].data[value] = torch.LongTensor(df[value].astype('category').cat.codes.values)

    for key, (df, features) in edge_dict.items():
        for value in features:
            g.edges[key].data[value] = torch.LongTensor(df[value].values.astype(np.float32))

    # 실제 ID와 카테고리 ID 딕셔너리
    user_cat = users['userID'].astype('category').cat.codes.values
    item_cat = items['item_id'].astype('category').cat.codes.values

    user_cat_dict = {k: v for k, v in zip(user_cat, users['userID'].values)}
    item_cat_dict = {k: v for k, v in zip(item_cat, items['item_id'].values)}

    # Label
    val_dict = defaultdict(set)
    for userID, df in new_ratings.groupby('userID'):
        temp = df[df['timestamp'] == 1]
        val_dict[userID] = set(df[df['timestamp'] == 1]['item_id'].values)

    # Label
    te_dict = defaultdict(set)
    for userID, df in new_ratings.groupby('userID'):
        temp = df[df['timestamp'] == 2]
        te_dict[userID] = set(df[df['timestamp'] == 2]['item_id'].values)

    # Build title set
    textual_feature = {
        "season" : items["season"].values,
        "tpo" : items["tpo"].values,
        "fit" : items["fit"].values,
        "brightness" : items["brightness"].values,
        "temperature" : items["temperature"].values,
        "weight" : items["weight"].values,
    }

    # Dump the graph and the datasets
    dataset = {
        'train-graph': g,
        'user-data': users,
        'item-data': items, 
        'rating-data': new_ratings,
        'val-matrix': None,
        'test-matrix': torch.LongTensor([[0]]),
        'validset': val_dict,
        'testset': te_dict,
        'item-texts': textual_feature,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'item',
        'user-category': user_cat_dict,
        'item-category': item_cat_dict,
        'user-to-item-type': 'rated',
        'item-to-user-type': 'rated-by',
        'timestamp-edge-column': 'timestamp'}

    with open(args.output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print('Processing Completed!')


    
    

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default="KData")
    parser.add_argument('--output_path', type=str, default="graph_data/kdata_entire8.pkl")
    args = parser.parse_args()

    # Training
    build_graph(args)