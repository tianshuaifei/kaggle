# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
np.random.seed(2018)

from time import sleep, time

df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv('input/test.csv')
df_hist_trans = pd.read_csv('input/historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('input/new_merchant_transactions.csv')
print(df_train.shape,df_test.shape)

x1=pd.read_csv("stacking/submission_3.8356194719157974train.csv")
y1=pd.read_csv("stacking/submission_3.8356194719157974test.csv")
x1.rename(columns={'target':"target_1"},inplace=True)
y1.rename(columns={'target':"target_1"},inplace=True)
df_train=df_train.merge(x1,on='card_id',how='left')
df_test=df_test.merge(y1,on='card_id',how='left')


for df in [df_hist_trans,df_new_merchant_trans]:
    df['category_2'].fillna(1.0, inplace=True)
    df['category_3'].fillna('A', inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)


for df in [df_hist_trans,df_new_merchant_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['purchase_month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour

    df['day'] = df['purchase_date'].dt.day

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    #df['month_diff_1'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff_1'] = ((datetime.datetime(2018, 12, 28,22,25,00) - df['purchase_date']).dt.days) // 30
    df['month_diff'] = df['month_diff_1']+df['month_lag']
    # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

    df['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    df['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Childrens day: October 12 2017
    df['Children_day_2017'] = (pd.to_datetime('2017-10-12') - df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : 24th November 2017
    df['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Valentines Day
    df['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # 2018
    # Mothers Day: May 13 2018
    df['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - df['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
######################################################################################
    df['p_vs_m'] = df['purchase_amount'] / (df['month_lag'].abs() + 1)
    df['p_vs_i'] = df['purchase_amount'] / (df['installments'].abs() + 1)

#agg_hist=df_hist_trans.groupby("merchant_id").agg({"target":"mean"})

print(len(df_train),df_train.columns)
df_train['outliers'] = 0
df_train.loc[df_train['target'] < -30, 'outliers'] = 1
print(df_train['outliers'].value_counts())
##########################################################################################
df_new=df_hist_trans[["card_id","merchant_id"]].copy()
df_train_x=df_train[["card_id","target"]]

df_test["target"]=100
df_test_x=df_test[["card_id","target"]]

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_x, df_train["outliers"].values)):
    print("fold {}".format(fold_))
    train_x=df_train_x[["card_id","target"]].iloc[trn_idx].copy()
    val_x=df_train_x[["card_id","target"]].iloc[val_idx].copy()
    val_x=pd.concat([df_test_x,val_x])

    print(train_x.shape,val_x.shape,df_new.shape)
    df_x2=df_new[["card_id","merchant_id"]].merge(train_x,on='card_id',how='left')
    df_y2=df_new[["card_id","merchant_id"]].merge(val_x,on='card_id',how='left')
    print(df_x2.shape, df_y2.shape, df_new.shape)

    df_x2 = df_x2[df_x2['target'].notna()]
    df_y2 = df_y2[df_y2['target'].notna()]
    print(df_x2.shape, df_y2.shape, df_new.shape)
    order_label = df_x2.groupby("merchant_id")['target'].mean()
    df_y2["merchant_id_ctr"] = df_y2["merchant_id"].map(order_label)
    aggs={
        "merchant_id_ctr":["sum","mean"]
    }
    df_y2_agg=df_y2.groupby("card_id").agg(aggs)
    df_y2_agg.columns = [k + '_' + str(agg) for k in aggs.keys() for agg in aggs[k]]
    df_y2_agg.reset_index(drop=False, inplace=True)
    print(df_y2_agg.shape)
    df_train_x = df_train_x.merge(df_y2_agg[["card_id", "merchant_id_ctr_sum"]], on='card_id', how='left')
    df_test_x=df_test_x.merge(df_y2_agg[["card_id", "merchant_id_ctr_sum"]], on='card_id', how='left')
    df_train_x.fillna(0,inplace=True)
    df_test_x.fillna(0,inplace=True)
    if fold_>0:
        df_train_x["merchant_id_ctr_sum"] = df_train_x["merchant_id_ctr_sum_x"] + df_train_x["merchant_id_ctr_sum_y"]
        df_test_x["merchant_id_ctr_sum"] = df_test_x["merchant_id_ctr_sum_x"] + df_test_x["merchant_id_ctr_sum_y"]
        del df_train_x["merchant_id_ctr_sum_x"], df_train_x["merchant_id_ctr_sum_y"]
        del df_test_x["merchant_id_ctr_sum_x"], df_test_x["merchant_id_ctr_sum_y"]

    print(df_train_x.head())
    print(df_test_x.head())
print(df_train_x.columns.tolist())
print(df_train_x.head())
print(df_train_x["merchant_id_ctr_sum"].head(20))
df_test_x["merchant_id_ctr_sum"]=df_test_x["merchant_id_ctr_sum"]/5
print(df_test_x["merchant_id_ctr_sum"].head(20))

df_train["merchant_id_ctr_sum"]=df_train_x["merchant_id_ctr_sum"].values
df_test["merchant_id_ctr_sum"]=df_test_x["merchant_id_ctr_sum"].values
del df_train_x,df_test_x,df_new
gc.collect()
################################################################################
df_new=df_hist_trans[["card_id","merchant_id"]].copy()
df_train_x=df_train[["card_id","target"]]

df_test["target"]=100
df_test_x=df_test[["card_id","target"]]

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_x, df_train["outliers"].values)):
    print("fold {}".format(fold_))
    train_x=df_train_x[["card_id","target"]].iloc[trn_idx].copy()
    val_x=df_train_x[["card_id","target"]].iloc[val_idx].copy()
    val_x=pd.concat([df_test_x,val_x])

    print(train_x.shape,val_x.shape,df_new.shape)
    df_x2=df_new[["card_id","merchant_id"]].merge(train_x,on='card_id',how='left')
    df_y2=df_new[["card_id","merchant_id"]].merge(val_x,on='card_id',how='left')
    print(df_x2.shape, df_y2.shape, df_new.shape)

    df_x2 = df_x2[df_x2['target'].notna()]
    df_y2 = df_y2[df_y2['target'].notna()]
    print(df_x2.shape, df_y2.shape, df_new.shape)
    order_label = df_x2.groupby("merchant_id")['target'].mean()
    df_y2["merchant_id_ctr"] = df_y2["merchant_id"].map(order_label)
    aggs={
        "merchant_id_ctr":["sum","mean"]
    }
    df_y2_agg=df_y2.groupby("card_id").agg(aggs)
    df_y2_agg.columns = [k + '_' + str(agg) for k in aggs.keys() for agg in aggs[k]]
    df_y2_agg.reset_index(drop=False, inplace=True)
    print(df_y2_agg.shape)
    df_train_x = df_train_x.merge(df_y2_agg[["card_id", "merchant_id_ctr_mean"]], on='card_id', how='left')
    df_test_x=df_test_x.merge(df_y2_agg[["card_id", "merchant_id_ctr_mean"]], on='card_id', how='left')
    df_train_x.fillna(0,inplace=True)
    df_test_x.fillna(0,inplace=True)
    if fold_>0:
        df_train_x["merchant_id_ctr_mean"] = df_train_x["merchant_id_ctr_mean_x"] + df_train_x["merchant_id_ctr_mean_y"]
        df_test_x["merchant_id_ctr_mean"] = df_test_x["merchant_id_ctr_mean_x"] + df_test_x["merchant_id_ctr_mean_y"]
        del df_train_x["merchant_id_ctr_mean_x"], df_train_x["merchant_id_ctr_sum_y"]
        del df_test_x["merchant_id_ctr_mean_x"], df_test_x["merchant_id_ctr_sum_y"]

    print(df_train_x.head())
    print(df_test_x.head())
print(df_train_x.columns.tolist())
print(df_train_x.head())
print(df_train_x["merchant_id_ctr_mean"].head(20))
df_test_x["merchant_id_ctr_mean"]=df_test_x["merchant_id_ctr_mean"]/5
print(df_test_x["merchant_id_ctr_mean"].head(20))

df_train["merchant_id_ctr_mean"]=df_train_x["merchant_id_ctr_mean"].values
df_test["merchant_id_ctr_mean"]=df_test_x["merchant_id_ctr_mean"].values
del df_train_x,df_test_x,df_new
gc.collect()
##############################################################
# df_hist_trans=df_hist_trans.merge(df_train,on='card_id',how='left')
# df_new_merchant_trans=df_new_merchant_trans.merge(df_train,on='card_id',how='left')
# for f in ["merchant_id"]:
#     order_label = df_hist_trans.groupby([f])['target'].mean()
#     df_hist_trans[f+"_ctr"] = df_hist_trans[f].map(order_label)
# for f in ["merchant_id"]:
#     order_label = df_new_merchant_trans.groupby([f])['target'].mean()
#     df_new_merchant_trans[f+"_ctr"] = df_new_merchant_trans[f].map(order_label)
#
# print(df_hist_trans["merchant_id_ctr"])
# df_hist_trans["card_id_time_merchant_category_id"]=df_hist_trans["card_id"]+"_"+df_hist_trans["merchant_category_id"].astype("str")
# dic=df_hist_trans["card_id_time_merchant_category_id"].value_counts().to_dict()
# df_hist_trans["card_id_time_merchant_category_id_count"]=df_hist_trans["card_id_time_merchant_category_id"].apply(lambda x:dic[x])

def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + str(agg) for k in aggs.keys() for agg in aggs[k]]

def make_agg_feature(df,name="hist"):
    aggs = {
        "city_id": ["nunique","count"],
        "state_id": ["nunique"],
        'weekofyear':['nunique'],
        'dayofweek': ['nunique'],
        #'year': ['nunique'],
        'purchase_month': ['nunique','mean', 'max', 'min', 'std'],
        'hour': ['nunique'],
        'day': ['nunique'],
        'subsector_id': ['nunique'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        "card_id":["size"],
        "category_1":['sum', 'mean'],
        "weekend": ['sum', 'mean'],
        "month_diff_1": ['mean'],
        "month_diff": ['mean'],
    }

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'median','mean', 'var', 'std']
    aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var']
    if name=="hist" or name =="new_hist":
        aggs['Christmas_Day_2017'] = ["mean"]
        aggs['fathers_day_2017'] = ["mean"]
        aggs['Children_day_2017'] = ["mean"]
        aggs['Black_Friday_2017'] = ["mean"]
        aggs['Valentine_day_2017'] = ["mean"]
        aggs['Mothers_Day_2018'] = ["mean"]

        aggs['p_vs_m']=['mean','std']
        aggs['p_vs_i']=['mean', 'std']
    # if name == "new_hist":
    #     aggs["merchant_id_ctr"]=["mean","sum"]

        # aggs['duration'] = ['mean', 'min', 'max']
        # aggs['amount_month_ratio'] = ['mean', 'min']


    for col in ['category_2', 'category_3']:
        df[col + '_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        aggs[col + '_mean'] = ['mean']

    new_columns = get_new_columns(name, aggs)
    df_group = df.groupby('card_id').agg(aggs)
    df_group.columns = new_columns
    df_group.reset_index(drop=False, inplace=True)
    df_group[name+'_purchase_date_diff'] = (df_group[name+'_purchase_date_max'] - df_group[name+'_purchase_date_min']).dt.days
    df_group[name+'_purchase_date_average'] = df_group[name+'_purchase_date_diff'] /df_group[name+'_card_id_size']
    df_group[name+'_purchase_date_uptonow'] = (datetime.datetime(2018, 12, 28,22,25,00) - df_group[name+'_purchase_date_max']).dt.days

    df_group[name + '_purchase_amount_average'] = df_group[name + '_purchase_amount_sum'] /df_group[name + '_card_id_size']
    df_group[name + '_purchase_amount_av'] = df_group[name + '_purchase_amount_sum'] /df_group[name + '_month_diff_mean']


    return df_group


authorized_transactions = df_hist_trans[df_hist_trans['authorized_flag'] == 1]
historical_transactions = df_hist_trans[df_hist_trans['authorized_flag'] == 0]

df_group=make_agg_feature(authorized_transactions,name="hist")
df_train = df_train.merge(df_group,on='card_id',how='left')
df_test = df_test.merge(df_group,on='card_id',how='left')
del df_group;gc.collect()

df_group=make_agg_feature(historical_transactions,name="hist_2_")
df_train = df_train.merge(df_group,on='card_id',how='left')
df_test = df_test.merge(df_group,on='card_id',how='left')
del df_group;gc.collect()

df_group=make_agg_feature(df_new_merchant_trans,name="new_hist")
df_train = df_train.merge(df_group, on='card_id', how='left')
df_test = df_test.merge(df_group, on='card_id', how='left')
del df_group;gc.collect()
################################################
def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])
    #grouped = history.groupby(['card_id', 'month_diff'])

    agg_func = {'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
                'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
                }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['month_lag_'+'_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    return final_group
final_group =  aggregate_per_month(authorized_transactions)
df_train = pd.merge(df_train, final_group, on='card_id', how='left')
df_test = pd.merge(df_test, final_group, on='card_id', how='left')
#####################################################################
def successive_aggregates(df, field1, field2):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['min', 'max'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    return u
additional_fields = successive_aggregates(df_new_merchant_trans, 'city_id', 'purchase_amount')
additional_fields = additional_fields.merge(successive_aggregates(df_new_merchant_trans, 'installments', 'purchase_amount'),
                                            on = 'card_id', how='left')
df_train = pd.merge(df_train, additional_fields, on='card_id', how='left')
df_test = pd.merge(df_test, additional_fields, on='card_id', how='left')
##################################################

#del df_hist_trans;gc.collect()
#del df_new_merchant_trans;gc.collect()


print(df_train.columns)

for df in [df_train,df_test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime(2018, 12, 28,22,25,00) - df['first_active_month']).dt.days
    df['elapsed_time']=df['elapsed_time']/30
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days

    for f in ['hist_purchase_date_max','hist_purchase_date_min',
              'new_hist_purchase_date_max','new_hist_purchase_date_min',
              'hist_2__purchase_date_max',
              'hist_2__purchase_date_min'
              ]:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    df['card_id_total_diff'] = df['new_hist_card_id_size']/df['card_id_total']
    df['card_id_day_avg'] = df['elapsed_time'] / df['card_id_total']
    df.drop(["hist_card_id_size","new_hist_card_id_size"],axis=1, inplace=True)
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_total_diff'] = df['new_hist_purchase_amount_sum']/df['purchase_amount_total']
    df['purchase_amount_total_diff2'] = df['hist_purchase_amount_sum'] - df['new_hist_purchase_amount_sum']
    df['purchase_amount_total_diff3'] = df['hist_purchase_amount_av'] - df['new_hist_purchase_amount_av']

    ###############################
    df["f1"]=df["new_hist_purchase_date_uptonow"]-df["hist_purchase_date_uptonow"]
    df["f2"] = df["new_hist_purchase_date_uptonow"]/df["hist_purchase_date_uptonow"]
    df["f3"]=df["new_hist_merchant_id_nunique"]/(df["hist_merchant_id_nunique"]+0.001)
    df["f4"] = df['new_hist_purchase_amount_mean']-df['hist_purchase_amount_mean']
    df["f5"] = df['new_hist_month_lag_mean']/(df['hist_month_lag_mean']+0.001)
    df["f6"] = df['new_hist_month_diff_mean'] / (df['hist_month_diff_mean'] + 0.001)

    df["t1"] = df["new_hist_month_diff_mean"] + df["hist_month_diff_mean"]
    df["t2"] = df["new_hist_purchase_date_uptonow"] + df["hist_purchase_date_uptonow"]

    #df['hist_sleep'] = df['hist_purchase_date_diff'] - df['hist_day_nunique']


for f in ['feature_1','feature_2','feature_3']:
    order_label = df_train.groupby([f])['outliers'].mean()
    df_train[f] = df_train[f].map(order_label)
    df_test[f] = df_test[f].map(order_label)

# for l in ["month","elapsed_time"]:
#     order_label = df_train.groupby([f])['target'].mean()
#     df_train[f+"_target"] = df_train[f].map(order_label)
#     df_test[f+"_target"] = df_test[f].map(order_label)
#print(df_train[["hist_merchant_id_ctr_sum","hist_merchant_id_ctr_mean","outliers","target"]])
df_train.to_csv("feature/tsf_train_feature.csv",index=False)
df_test.to_csv("feature/tsf_test_feature.csv",index=False)
#######################################################

# def label_category(arr):
#     return list(set(arr))
#
# def make_vec(df,name):
#     aggs={
#           "merchant_category_id": [label_category],
#           }
#     agg_new = df.groupby(['card_id']).agg(aggs)
#     agg_new.columns = [name + '_' + k + '_' + "label" for k in aggs.keys()]
#
#     agg_new.reset_index(drop=False, inplace=True)
#     print(agg_new.columns)
#     return agg_new
# df_hist_trans_label=make_vec(df_hist_trans,name="a")
# df_train = df_train.merge(df_hist_trans_label,on='card_id',how='left')
# df_test = df_test.merge(df_hist_trans_label,on='card_id',how='left')
#
# df_new_merchant_label=make_vec(df_new_merchant_trans,name="b")
# df_train = df_train.merge(df_new_merchant_label,on='card_id',how='left')
# df_test = df_test.merge(df_new_merchant_label,on='card_id',how='left')
#
#
# def getInterval(arrLike):
#     x_1 = arrLike['a_merchant_category_id_label']
#     x_2 = arrLike['b_merchant_category_id_label']
#     return len(list(set(x_1).intersection(set(x_2))))
#
#
# #
# df_train["m_c_count"]=df_train.apply(getInterval, axis = 1)
# df_test["m_c_count"]=df_test.apply(getInterval, axis = 1)
# print(df_train["m_c_count"].head())
# del df_train["a_merchant_category_id_label"],df_train["b_merchant_category_id_label"]
# del df_test["a_merchant_category_id_label"],df_test["b_merchant_category_id_label"]
###################################################################################
def label_str(arr):
    #return " ".join(str(i) for i in list(set(arr)))
    return " ".join(str(i) for i in list(arr))

def make_count_vec(df,name):
    aggs={"city_id": [label_str],
          "merchant_category_id": [label_str],
          'state_id': [label_str],
          'installments':[label_str],
          "subsector_id": [label_str],}
    agg_new = df.groupby(['card_id']).agg(aggs)
    agg_new.columns = [name + '_' + k + '_' + "label" for k in aggs.keys()]
    agg_new.reset_index(drop=False, inplace=True)
    return agg_new

df_hist_trans_label=make_count_vec(df_hist_trans,name="hist")
df_train = df_train.merge(df_hist_trans_label,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_label,on='card_id',how='left')

###########################################################################################
# 默认加载 如果 增加了cate类别特征 请改成false重新生成
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import os
hist_label_feature=["hist_installments_label",'hist_city_id_label', 'hist_merchant_category_id_label','hist_state_id_label', 'hist_subsector_id_label']
if os.path.exists('tsf_base_train_csr.npz') and True:
    print('load_csr---------')
    base_train_csr = sparse.load_npz('tsf_base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz('tsf_base_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(df_train), 0))
    base_predict_csr = sparse.csr_matrix((len(df_test), 0))
    cv = CountVectorizer()
    for feature in hist_label_feature:
        df_hist_trans_label[feature] = df_hist_trans_label[feature].astype(str)
        cv.fit(df_hist_trans_label[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(df_train[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(df_test[feature].astype(str))), 'csr','bool')
    print('cv prepared !')
    sparse.save_npz('tsf_base_train_csr.npz', base_train_csr)
    sparse.save_npz('tsf_base_predict_csr.npz', base_predict_csr)
####################################################################################################

###########################################################################################

df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']+hist_label_feature]
target = df_train['target']
train_y=df_train['target'].values
del df_train['target']
print(len(df_train_columns))
print(df_train_columns)

train_csr = sparse.hstack((sparse.csr_matrix(df_train[df_train_columns]), base_train_csr), 'csr').astype('float32')
predict_csr = sparse.hstack((sparse.csr_matrix(df_test[df_train_columns]), base_predict_csr), 'csr').astype('float32')
print(train_csr.shape)

param = {'num_leaves': 64,
         'min_data_in_leaf': 30,  # 'objective':'regression',
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 16,
         "random_state": 4590
         }
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()
cv_scores = []
cv_rounds = []
clf_name="tsf_lgb"
for fold_, (train_index, val_index) in enumerate(folds.split(train_csr, df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_csr[train_index],
                           label=train_y[train_index])  # , categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_csr[val_index],
                           label=train_y[val_index])  # , categorical_feature=categorical_feats)

    clf = lgb.train(param, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=200)
    oof[val_index] = clf.predict(train_csr[val_index], num_iteration=clf.best_iteration)

    predictions += clf.predict(predict_csr, num_iteration=clf.best_iteration) / folds.n_splits

    cv_scores.append(mean_squared_error(train_y[val_index], oof[val_index]) ** 0.5)
    cv_rounds.append(clf.best_iteration)

    print("%s now score is:" % clf_name, cv_scores)
    print("%s now round is:" % clf_name, cv_rounds)

r=np.sqrt(mean_squared_error(oof, target))
print("%s all score is:" % clf_name, r)
with open("a_lgb_score_cv.txt", "a") as f:
    f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
    f.write("%s now round is:" % clf_name + str(cv_rounds) + "\n")
    f.write("%s_score_mean:" % clf_name + str(np.mean(cv_scores)) + "\n")
    f.write("%s_score_std:" % clf_name + str(np.std(cv_scores)) + "\n")
    f.write("lgb all score is:" + str(r) + "\n" + "***" + "\n")

sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("stacking/submission_"+str(r)+"_test.csv", index=False)

sub_df = pd.DataFrame({"card_id":df_train["card_id"].values})
sub_df["target"] = oof
sub_df.to_csv("stacking/submission_"+str(r)+"_train.csv", index=False)
