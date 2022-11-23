import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import random
import argparse


# 데이터 크기 확인 함수
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

## 타입별 평균 크기 확인 함수
def type_memory(data) :
    for dtype in ['float','int','object']:
        selected_dtype = data.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

## 이산형 데이터 사이즈 축소 함소
def int_memory_reduce(data) :
    data_int = data.select_dtypes(include=['int'])
    converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')
    print(f"Before : {mem_usage(data_int)} -> After : {mem_usage(converted_int)}")
    data[converted_int.columns] = converted_int
    return data

## 연속형 데이터 사이즈 축소 함소
def float_memory_reduce(data) :
    data_float = data.select_dtypes(include=['float'])
    converted_float = data_float.apply(pd.to_numeric,downcast='float')
    print(f"Before : {mem_usage(data_float)} -> After : {mem_usage(converted_float)}")
    data[converted_float.columns] = converted_float
    return data

## 문자형 데이터 사이즈 축소 함소
def object_memory_reduce(data) :
    gl_obj = data.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:,col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:,col] = gl_obj[col]
    print(f"Before : {mem_usage(gl_obj)} -> After : {mem_usage(converted_obj)}")
    data[converted_obj.columns] = converted_obj
    return data

def load_dataset(args):
    kdf = pd.read_excel(args.dataset_path, sheet_name='rawdata_2차(1118)', engine='openpyxl')
    #kdf = kdf.rename(columns=kdf.iloc[0])
    #kdf = kdf.drop(kdf.index[0])
    print(f"Original Dataset Shape : {kdf.shape}")
    return kdf


def generate_userdata(args, kdf):
    #### Generating User Dataset
    user_data = kdf[["R_id", "r_gender", "age", "mar", "job", "income", 
                     "r_style1", "r_style2", "r_style3", "r_style4", "r_style5"]]
    user_data = user_data.drop_duplicates(['R_id']).reset_index(drop=True)
    user_data["user"] = user_data.reset_index()["index"]
    user_data = user_data[["user", "R_id", "r_gender", "age", "mar", "job", "income", 
                     "r_style1", "r_style2", "r_style3", "r_style4", "r_style5"]]

    user_data.columns = ["user", "user_name", "r_gender", "age", "mar", "job", "income",
                         "r_style1", "r_style2", "r_style3", "r_style4", "r_style5"]
    user_data = user_data[["user", "user_name", "r_gender", "age", "mar", "job", "income",
                         "r_style1", "r_style2", "r_style3", "r_style4", "r_style5"]]
    user_data = int_memory_reduce(user_data)
    user_data = float_memory_reduce(user_data)
    user_data = object_memory_reduce(user_data)
    user_data.to_csv(f"{args.save_path}/user_data.csv")
    print(f"User Dataset Shape : {user_data.shape}")
    return user_data


def generate_itemdata(args, kdf):
    #### Generating Item Dataset
    # Q2
    q2_dict = {"1" : "spring fall", "2" : "summer", "3" : "winter"}
    # Q3
    q3_dict = {"1" : "attendance",
               "2" : "date",
               "3" : "event",
               "4" : "social gathering",
               "5" : "daily",
               "6" : "leisure sports",
               "7" : "trip vacation",
               "8" : "etc"}
    # Q411
    q411_dict = {"1":"loose","2":"appropriate","3":"tight"}
    # Q412
    q412_dict = {"1":"dark","2":"bright"}
    # Q413
    q413_dict = {"1":"cold","2":"warm"}
    # Q414
    q414_dict = {"1":"heavy","2":"light"}
    # Q4201
    q4201_dict = {"0":"no","1":"nice"}
    # Q4202
    q4202_dict = {"0":"no","2":"urban"}
    # Q4203
    q4203_dict = {"0":"no","3":"trendy"}
    # Q4204
    q4204_dict = {"0":"no","4":"sophisticated"}
    # Q4205
    q4205_dict = {"0":"no","5":"clean"}
    # Q4206
    q4206_dict = {"0":"no","6":"magnificent"}
    # Q4207
    q4207_dict = {"0":"no","7":"unique"}
    # Q4208
    q4208_dict = {"0":"no","8":"easy"}
    # Q4209
    q4209_dict = {"0":"no","9":"open mined"}
    # Q4210
    q4210_dict = {"0":"no","10":"practical"}
    # Q4211
    q4211_dict = {"0":"no","11":"activity"}
    # Q4212
    q4212_dict = {"0":"no","12":"comfortable"}
    # Q4213
    q4213_dict = {"0":"no","13":"bubbly"}
    # Q4214
    q4214_dict = {"0":"no","14":"feminine"}
    # Q4215
    q4215_dict = {"0":"no","15":"manly"}
    # Q4216
    q4216_dict = {"0":"no","16":"soft"}


    item_data = kdf[['imgName','era','style','gender', 
                     'Q1','Q2','Q3','Q411','Q412','Q413','Q414','Q4201','Q4202','Q4203','Q4204','Q4205',
                     'Q4206','Q4207','Q4208','Q4209','Q4210','Q4211','Q4212','Q4213','Q4214','Q4215','Q4216','Q5']]
    
    item_data['Q414'] = item_data['Q414'].fillna(item_data['Q414'].mode()[0])
    
    for col_i in item_data.columns.tolist()[4:]:
        item_data.loc[:, col_i] = item_data.loc[:, col_i].astype("int")
    
    for col_i in ['Q2','Q3','Q411','Q412','Q413','Q414','Q4201','Q4202','Q4203','Q4204','Q4205',
              'Q4206','Q4207','Q4208','Q4209','Q4210','Q4211','Q4212','Q4213','Q4214','Q4215','Q4216']:
        item_data.loc[:, col_i] = item_data.loc[:, col_i].astype("str")

    item_data["season"] = [q2_dict[str(i)] for i in item_data["Q2"].values.tolist()]
    item_data["tpo"] = [q3_dict[str(i)] for i in item_data["Q3"].values.tolist()]
    item_data["fit"] = [q411_dict[str(i)] for i in item_data["Q411"].values.tolist()]
    item_data["brightness"] = [q412_dict[str(i)] for i in item_data["Q412"].values.tolist()]
    item_data["temperature"] = [q413_dict[str(i)] for i in item_data["Q413"].values.tolist()]
    item_data["weight"] = [q414_dict[str(i)] for i in item_data["Q414"].values.tolist()]

    item_data["nice"] = [q4201_dict[str(i)] for i in item_data["Q4201"].values.tolist()]
    item_data["urban"] = [q4202_dict[str(i)] for i in item_data["Q4202"].values.tolist()]
    item_data["trendy"] = [q4203_dict[str(i)] for i in item_data["Q4203"].values.tolist()]
    item_data["sophisticated"] = [q4204_dict[str(i)] for i in item_data["Q4204"].values.tolist()]
    item_data["clean"] = [q4205_dict[str(i)] for i in item_data["Q4205"].values.tolist()]
    item_data["magnificent"] = [q4206_dict[str(i)] for i in item_data["Q4206"].values.tolist()]
    item_data["unique"] = [q4207_dict[str(i)] for i in item_data["Q4207"].values.tolist()]
    item_data["easy"] = [q4208_dict[str(i)] for i in item_data["Q4208"].values.tolist()]
    item_data["open"] = [q4209_dict[str(i)] for i in item_data["Q4209"].values.tolist()]
    item_data["practical"] = [q4210_dict[str(i)] for i in item_data["Q4210"].values.tolist()]
    item_data["activity"] = [q4211_dict[str(i)] for i in item_data["Q4211"].values.tolist()]
    item_data["comfortable"] = [q4212_dict[str(i)] for i in item_data["Q4212"].values.tolist()]
    item_data["bubbly"] = [q4213_dict[str(i)] for i in item_data["Q4213"].values.tolist()]
    item_data["feminine"] = [q4214_dict[str(i)] for i in item_data["Q4214"].values.tolist()]
    item_data["manly"] = [q4215_dict[str(i)] for i in item_data["Q4215"].values.tolist()]
    item_data["soft"] = [q4216_dict[str(i)] for i in item_data["Q4216"].values.tolist()]

    item_data = item_data[["imgName", "era", "style", "gender", 'season', 'tpo', 'fit', 'brightness',
                           'temperature', 'weight', 'nice', 'urban', 'trendy', 'sophisticated',
                           'clean', 'magnificent', 'unique', 'easy', 'open', 'practical',
                           'activity', 'comfortable', 'bubbly', 'feminine', 'manly', 'soft', "Q1", "Q5"]]

    item_data = pd.get_dummies(item_data, columns = ['season', "tpo", 'fit', 'brightness',
                                                     'temperature', 'weight', 'nice', 'urban', 'trendy', 'sophisticated',
                                                     'clean', 'magnificent', 'unique', 'easy', 'open', 'practical',
                                                     'activity', 'comfortable', 'bubbly', 'feminine', 'manly', 'soft'])
    item_data = item_data.groupby(["imgName", "era", "style", "gender"]).sum().reset_index()

    item_data.columns = ["item_name", "era", "style", "gender"] + item_data.columns.tolist()[4:]
    item_data = int_memory_reduce(item_data)
    item_data = float_memory_reduce(item_data)
    item_data = object_memory_reduce(item_data)

    # season
    spring_and_fall = item_data[item_data["season_spring fall"]>=1].index
    summer = item_data[item_data["season_summer"]>=1].index
    winter = item_data[item_data["season_winter"]>=1].index

    # tpo
    attend = item_data[item_data["tpo_attendance"]>=1].index
    daily = item_data[item_data["tpo_daily"]>=1].index
    date = item_data[item_data["tpo_date"]>=1].index
    etc = item_data[item_data["tpo_etc"]>=1].index
    event = item_data[item_data["tpo_event"]>=1].index
    sports = item_data[item_data["tpo_leisure sports"]>=1].index
    social = item_data[item_data["tpo_social gathering"]>=1].index
    trip = item_data[item_data["tpo_trip vacation"]>=1].index

    # fit
    appro = item_data[item_data["fit_appropriate"]>=1].index
    loo = item_data[item_data["fit_loose"]>=1].index
    tig = item_data[item_data["fit_tight"]>=1].index

    # brightness
    bright = item_data[item_data["brightness_bright"]>=1].index
    dark = item_data[item_data["brightness_dark"]>=1].index

    # temperature
    cold = item_data[item_data["temperature_cold"]>=1].index
    warm = item_data[item_data["temperature_warm"]>=1].index

    # weight
    heavy = item_data[item_data["weight_heavy"]>=1].index
    light = item_data[item_data["weight_light"]>=1].index



    nice_no = item_data[item_data["nice_no"]>=1].index
    nice_nice = item_data[item_data["nice_nice"]>=1].index

    urban_no = item_data[item_data["urban_no"]>=1].index
    urban_urban = item_data[item_data["urban_urban"]>=1].index

    trendy_no = item_data[item_data["trendy_no"]>=1].index
    trendy_trendy = item_data[item_data["trendy_trendy"]>=1].index

    sophisticated_no = item_data[item_data["sophisticated_no"]>=1].index
    sophisticated_sohp = item_data[item_data["sophisticated_sophisticated"]>=1].index

    clean_no = item_data[item_data["clean_no"]>=1].index
    clean_clean = item_data[item_data["clean_clean"]>=1].index

    magnificent_no = item_data[item_data["magnificent_no"]>=1].index
    magnificent_magnificent = item_data[item_data["magnificent_magnificent"]>=1].index

    unique_no = item_data[item_data["unique_no"]>=1].index
    unique_unique = item_data[item_data["unique_unique"]>=1].index

    easy_no = item_data[item_data["easy_no"]>=1].index
    easy_easy = item_data[item_data["easy_easy"]>=1].index

    open_no = item_data[item_data["open_no"]>=1].index
    open_open = item_data[item_data["open_open mined"]>=1].index

    practical_no = item_data[item_data["practical_no"]>=1].index
    practical_practical = item_data[item_data["practical_practical"]>=1].index

    activity_no = item_data[item_data["activity_no"]>=1].index
    activity_activity = item_data[item_data["activity_activity"]>=1].index

    comfortable_no = item_data[item_data["comfortable_no"]>=1].index
    comfortable_comfortable = item_data[item_data["comfortable_comfortable"]>=1].index

    bubbly_no = item_data[item_data["bubbly_no"]>=1].index
    bubbly_bubbly = item_data[item_data["bubbly_bubbly"]>=1].index

    feminine_no = item_data[item_data["feminine_no"]>=1].index
    feminine_feminine = item_data[item_data["feminine_feminine"]>=1].index

    manly_no = item_data[item_data["manly_no"]>=1].index
    manly_manly = item_data[item_data["manly_manly"]>=1].index

    soft_no = item_data[item_data["soft_no"]>=1].index
    soft_soft = item_data[item_data["soft_soft"]>=1].index

    item_data["season"] = "" 
    item_data["tpo"] = ""
    item_data["fit"] = ""
    item_data["brightness"] = ""
    item_data["temperature"] = ""
    item_data["weight"] = ""

    item_data.iloc[spring_and_fall, -6] = "spring fall"
    item_data.iloc[summer, -6] = item_data.iloc[summer, -6] + " summer"
    item_data.iloc[winter, -6] = item_data.iloc[winter, -6] + " winter"
    item_data.iloc[attend, -5] = "attendance"
    item_data.iloc[daily, -5] = item_data.iloc[daily, -5] + " daily"
    item_data.iloc[date, -5] = item_data.iloc[date, -5] + " date"
    item_data.iloc[etc, -5] = item_data.iloc[etc, -5] + " etc"
    item_data.iloc[event, -5] = item_data.iloc[event, -5] + " event"
    item_data.iloc[sports, -5] = item_data.iloc[sports, -5] + " leisure sports"
    item_data.iloc[social, -5] = item_data.iloc[social, -5] + " social gathering"
    item_data.iloc[trip, -5] = item_data.iloc[trip, -5] + " social trip vacation"
    item_data.iloc[appro, -4] = "appropriate"
    item_data.iloc[loo, -4] = item_data.iloc[loo, -4] + " loose"
    item_data.iloc[tig, -4] = item_data.iloc[tig, -4] + " tight"
    item_data.iloc[bright, -3] = item_data.iloc[bright, -3] + " bright"
    item_data.iloc[dark, -3] = item_data.iloc[dark, -3] + " dark"
    item_data.iloc[cold, -2] = item_data.iloc[cold, -2] + " cold"
    item_data.iloc[warm, -2] = item_data.iloc[warm, -2] + " warm"
    item_data.iloc[heavy, -1] = item_data.iloc[heavy, -1] + " heavy"
    item_data.iloc[light, -1] = item_data.iloc[light, -1] + " light"
    item_data["item"] = item_data.reset_index()["index"]
    item_data = \
    item_data[['item','item_name', 'era', 'style', 'gender',
               'season', 'tpo', 'fit', 'brightness', 'temperature', 'weight',
               'nice_nice', 'nice_no', 'urban_no', 'urban_urban', 'trendy_no',
               'trendy_trendy', 'sophisticated_no', 'sophisticated_sophisticated',
               'clean_clean', 'clean_no', 'magnificent_magnificent', 'magnificent_no',
               'unique_no', 'unique_unique', 'easy_easy', 'easy_no', 'open_no',
               'open_open mined', 'practical_no', 'practical_practical',
               'activity_activity', 'activity_no', 'comfortable_comfortable',
               'comfortable_no', 'bubbly_bubbly', 'bubbly_no', 'feminine_feminine',
               'feminine_no', 'manly_manly', 'manly_no', 'soft_no', 'soft_soft']]
    item_data['season'] = item_data['season'].str.strip()
    item_data['tpo'] = item_data['tpo'].str.strip()
    item_data['fit'] = item_data['fit'].str.strip()
    item_data['brightness'] = item_data['brightness'].str.strip()
    item_data['temperature'] = item_data['temperature'].str.strip()
    item_data['weight'] = item_data['weight'].str.strip()

    for re_col in item_data.columns.tolist()[11:]:
        item_data.loc[item_data[re_col] >= 1, re_col] = 1

    item_data = int_memory_reduce(item_data)
    item_data = float_memory_reduce(item_data)
    item_data = object_memory_reduce(item_data)
    item_data.to_csv(f"{args.save_path}/item_data.csv")
    print(f"Item Dataset Shape : {item_data.shape}")
    return item_data


def generate_ratedata(args, kdf, user_data, item_data):
    rate_data = kdf[["R_id", "imgName", "Q1", "Q5"]]
    rate_data.columns = ["user", "item", "Q1", "Q5"]

    u_dict = user_data[["user", "user_name"]].astype("str").set_index("user_name")
    u_dict = u_dict.T.to_dict('records')[0]

    i_dict = item_data[["item", "item_name"]].astype("str").set_index("item_name")
    i_dict = i_dict.T.to_dict('records')[0]

    #rate_data["user"] = [int(u_dict[str(i)]) for i in rate_data["user"].values.tolist()]
    rate_data["item"] = [int(i_dict[str(i)]) for i in rate_data["item"].values.tolist()]

    rate_data_q1q2 = rate_data[["user", "item", "Q1", "Q5"]]
    rate_data_q1q2["rate"] = rate_data_q1q2[["Q1", "Q5"]].mean(axis=1)
    rate_data_q1q2 = rate_data_q1q2[["user", "item", "rate"]]
    rate_data_q1q2 = int_memory_reduce(rate_data_q1q2)
    rate_data_q1q2 = float_memory_reduce(rate_data_q1q2)
    rate_data_q1q2.to_csv(f"{args.save_path}/rate_data.csv")
    print(f"Rating Dataset Shape : {rate_data_q1q2.shape}")
    

def build_data(args):
    kdf = load_dataset(args)
    user_data = generate_userdata(args, kdf)
    item_data = generate_itemdata(args, kdf)
    generate_ratedata(args, kdf, user_data, item_data)
    print('\033[95m' + f'----complete----' + '\033[0m')
    
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="KData/패션 선호도조사 결과_중간전달 data(1118).xlsx")
    parser.add_argument('--save-path', type=str, default='KData')
    args = parser.parse_args()

    # Training
    build_data(args)