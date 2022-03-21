import gc
import time
from zipfile import ZipFile
from urllib.request import urlretrieve
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

pd.set_option('max_column', 500)
pd.set_option('max_row', 500)


df_chunks = pd.read_csv('dataset/tbrain_cc_training_48tags_hash_final.csv', chunksize=100000,
                        dtype=DTYPES,
                        usecols=USE_COLS
)

df_list = []
for i, df_chunk in enumerate(df_chunks):
    print(f'\r{i}', end="")
    df_list.append(df_chunk)
    
df_all = pd.concat(df_list)
df_all.shape

PT_PCT_FEATURES = ['domestic_offline_amt_pct', 'domestic_online_amt_pct', 'overseas_offline_amt_pct', 'overseas_online_amt_pct']
PT_CNT_FEATURES = ['domestic_offline_cnt', 'domestic_online_cnt', 'overseas_offline_cnt', 'overseas_online_cnt']

# Drop samples witch didn't bought anythings and refund noly
mask = (df_all[PT_CNT_FEATURES] <= 0).all(axis=1)
drop_idx_cnt_zero = df_all[mask].index
df_all.drop(drop_idx_cnt_zero, axis=0, inplace=True)
df_all.reset_index(drop=True, inplace=True)
print(f"Drop {len(drop_idx_cnt_zero)} samples.")

# Power transfomr to deal with extreme value
df_all.slam = df_all.slam.apply(lambda x: np.nan if isinstance(x, str) else x)
df_all.slam = df_all.slam.apply(np.log1p)
df_all.txn_amt = df_all.txn_amt.apply(np.log1p)
for column in df_all.columns:
    df_all[column].fillna(df_all[column].mode()[0], inplace=True)
    
# Bucketize txn_amt
df_all.txn_amt = pd.cut(df_all.txn_amt, 100000, labels=False, duplicates='drop')
df_all.txn_amt = df_all.txn_amt.astype('str')

# Bucketize slam
df_all.slam = pd.qcut(df_all.slam, 30, labels=False, duplicates='drop')
df_all.shop_tag = df_all.shop_tag.replace('other', '49')
df_all.dt = df_all.dt.astype('str')
df_all.txn_cnt = df_all.txn_cnt.astype('str')

# Get user's info and reduce memory usage
user_group = df_all.groupby('chid')
user_data = user_group.tail(1).reset_index()[['chid','masts', 'educd', 'trdtp', 'naty', 'poscd', 'cuorg', 'slam', 'gender_code', 'age']]
del df_all

for col in ['masts', 'educd', 'trdtp', 'naty', 'poscd', 'cuorg', 'slam', 'gender_code', 'age']:
    user_data[col] = user_data[col].apply(lambda x: x if isinstance(x, (float, int, np.int8, np.float16)) else x[-1])
    
user_data = user_data.astype(float).astype(int).astype(str)
for name, col in user_data.iteritems():
    if name == 'chid':
        continue
    user_data[name] = pd.Categorical(user_data[name])
    user_data[name] = user_data[name].cat.codes
    
# Get 
shop_data = pd.DataFrame(
    data={
            "chid": list(user_group.groups.keys()),
            "month": list(user_group.dt.apply(list)),
            "shop_tag": list(user_group.shop_tag.apply(list)),
            "txn_cnt": list(user_group.txn_cnt.apply(list)),
            "txn_amt": list(user_group.txn_amt.apply(list)),
    }
)

time_tail = 24
windows_length = 13
step_size = 1

def slide_windows(value, position, win_len, step_size, time_tail, get_position=False):
    seq_list = []
    pos_list = []
    start_pos = 0
    while True:
        pos = []
        seq = []
        value_zip = zip(position, value)
        end_pos = start_pos + win_len
        if end_pos > time_tail:
            break
        for p, v in value_zip:
            if start_pos < int(p) <= end_pos:
                pos.append(p)
                seq.append(v)
        idx = [str(i) for i in np.arange(start_pos+1, end_pos+1)]
        lacks = list(set(idx).difference(pos))
        if lacks:
            pos.extend(lacks)
            seq.extend(['0' for _ in lacks])
            zipped_lists = zip(pos, seq)
            sorted_pairs = sorted(zipped_lists, key=lambda x: int(x[0]))
            tuples = zip(*sorted_pairs)
            pos, seq = [list(tuple) for tuple in tuples]
        pos_list.append(pos)
        seq_list.append(seq)
        start_pos += step_size
    if get_position:
        return pos_list
    return seq_list

def slider(df):
    length = len(df)
    spacer = int(length // 2)
    dfs = [df.iloc[:spacer], df.iloc[spacer:]]
    result_dfs = []
    for df_split in dfs:
        print("Processing slide windows...")
        print("  position...", end="")
        df_split['position'] = df_split[['month', 'shop_tag',]].apply(
            lambda uid: slide_windows(uid.shop_tag, uid.month, windows_length, step_size, time_tail, True), axis=1
            )
        print(" Done")

        print("  txn_amt...", end="")
        df_split['txn_amt'] = df_split[['month', 'txn_amt',]].apply(
            lambda uid: slide_windows(uid.txn_amt, uid.month, windows_length, step_size, time_tail), axis=1
            )
        print(" Done")

        print("  shop_tag...", end="")
        df_split['shop_tag'] = df_split[['month', 'shop_tag',]].apply(
            lambda uid: slide_windows(uid.shop_tag, uid.month, windows_length, step_size, time_tail), axis=1
            )
        print(" Done")

        print("  txn_cnt...", end="")
        df_split['txn_cnt'] = df_split[['month', 'txn_cnt',]].apply(
            lambda uid: slide_windows(uid.txn_cnt, uid.month, windows_length, step_size, time_tail), axis=1
            )
        print(" Done")
        result_dfs.append(df_split)
    return pd.concat(result_dfs)

shop_data = slider(shop_data)

shop_data_tag = shop_data[['chid', 'shop_tag']].explode('shop_tag', ignore_index=True)
shop_data_pos = shop_data[['position']].explode('position', ignore_index=True)
shop_data_cnt = shop_data[['txn_cnt']].explode('txn_cnt', ignore_index=True)
shop_data_amt = shop_data[['txn_amt']].explode('txn_amt', ignore_index=True)
shop_data_transformed = pd.concat([shop_data_tag, shop_data_pos, shop_data_cnt, shop_data_amt], axis=1)
del shop_data_tag, shop_data_cnt, shop_data_amt, shop_data
gc.collect()

shop_data_transformed = shop_data_transformed.join(user_data.astype(int).set_index('chid'), on='chid')
shop_data_transformed.shop_tag = shop_data_transformed.shop_tag.apply(
    lambda x: ','.join(x)
    )
shop_data_transformed.txn_cnt = shop_data_transformed.txn_cnt.apply(
    lambda x: ','.join(x)
    )
shop_data_transformed.txn_amt = shop_data_transformed.txn_amt.apply(
    lambda x: ','.join(x)
    )
shop_data_transformed.position = shop_data_transformed.position.apply(
    lambda x: ','.join(x)
    )

shop_data_transformed.rename(
    columns={'chid':'user_id',
             'shop_tag': 'sequence_shop_tag', 
             'position': 'sequence_month',
             'txn_cnt': 'sequence_count',
             'txn_amt': 'sequence_amount',
             'masts':'marriage',
             'educd':'education',
             'trdtp':'occupation',
             'naty':'country',
             'poscd':'position',
             'cuorg':'source',
             'slam':'quota',
             'gender_code':'gender'}, inplace=True)

random_selection = np.random.rand(len(purchase_sequence_and_profile.index)) <= 0.8
train_data = purchase_sequence_and_profile[random_selection]
test_data = purchase_sequence_and_profile[~random_selection]
print('train_data size: ', train_data.shape)
print('test_data size: ', test_data.shape)
train_data.to_csv("dataset/train_data_purchase_feature.csv", index=False, sep=",")
test_data.to_csv("dataset/test_data_purchase_feature.csv", index=False, sep=",")