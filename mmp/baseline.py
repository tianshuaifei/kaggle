#https://www.kaggle.com/bogorodvo/upd-lightgbm-baseline-model-using-sparse-matrix
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
#import xgboost as xgb
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import metrics

import gc
gc.enable()

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
#
nrow=None
print('Download Train and Test Data.\n')
df_train = pd.read_csv('input/train.csv', dtype=dtypes, low_memory=True,nrows=nrow)
#train['MachineIdentifier'] = train.index.astype('uint32')
df_test  = pd.read_csv('input/test.csv',  dtype=dtypes, low_memory=True,nrows=nrow)
#test['MachineIdentifier']  = test.index.astype('uint32')
df_test["HasDetections"]=100
gc.collect()
print(df_train.shape,df_test.shape)

trans_dict = {
    'off': 'Off', '&#x02;': '2', '&#x01;': '1', 'on': 'On', 'requireadmin': 'RequireAdmin', 'OFF': 'Off',
    'Promt': 'Prompt', 'requireAdmin': 'RequireAdmin', 'prompt': 'Prompt', 'warn': 'Warn',
    '00000000': '0', '&#x03;': '3', np.nan: 'NoExist'
}
df_train.replace({'SmartScreen': trans_dict}, inplace=True)
df_test.replace({'SmartScreen': trans_dict}, inplace=True)

df_train.replace({'OrganizationIdentifier': {np.nan: 0}}, inplace=True)
df_test.replace({'OrganizationIdentifier': {np.nan: 0}}, inplace=True)

data=pd.concat([df_train,df_test])

id_columns=["DefaultBrowsersIdentifier",
            "AVProductStatesIdentifier",
            "CountryIdentifier",
            "CityIdentifier",
            "OrganizationIdentifier",
            "GeoNameIdentifier",
            "LocaleEnglishNameIdentifier",
            "IeVerIdentifier",
            "Census_OEMNameIdentifier",
            "Census_OEMModelIdentifier",
            "Census_ProcessorManufacturerIdentifier",
            "Census_ProcessorModelIdentifier",
            "Census_ProcessorClass",
            "Census_OSInstallLanguageIdentifier",
            "Census_OSUILocaleIdentifier",
            "Census_FirmwareManufacturerIdentifier",
            "Census_FirmwareVersionIdentifier",
            "Wdft_RegionIdentifier",
            ]
#data[id_columns].fillna("id_no",inplace=True)
from sklearn.preprocessing import LabelEncoder
for id_col in id_columns:
    lab = LabelEncoder()
    data[id_col]=lab.fit_transform(data[id_col].astype("str"))

cate_columns=["Platform",
              "ProductName",
              "Processor",
              "OsVer",
              "OsBuild",
              "OsSuite",
              "OsPlatformSubRelease",
              "OsBuildLab",
              "SkuEdition",
              "SmartScreen",
              "Census_MDC2FormFactor",
              "Census_DeviceFamily",
              "Census_PrimaryDiskTypeName",
              "Census_ChassisTypeName",
              "Census_PowerPlatformRoleName",
              "Census_InternalBatteryType",
              "Census_InternalBatteryNumberOfCharges",
              "Census_OSArchitecture",
              "Census_OSBranch",
              "Census_OSBuildRevision",
              "Census_OSBuildNumber",
              "Census_OSEdition",
              "Census_OSSkuName",
              "Census_OSInstallTypeName",
              "Census_OSWUAutoUpdateOptionsName",
              "Census_FlightRing",
              ]

for id_col in cate_columns:
    lab = LabelEncoder()
    data[id_col]=lab.fit_transform(data[id_col].astype("str"))

num_columns=["RtpStateBitfield",
             "AVProductsInstalled",
             "AVProductsEnabled",
             "AutoSampleOptIn",
             #"PuaMode",
             "SMode",
             "Census_ProcessorCoreCount",
             "Census_PrimaryDiskTotalCapacity",
             "Census_SystemVolumeTotalCapacity",
             "Census_TotalPhysicalRAM",
             "Census_InternalPrimaryDiagonalDisplaySizeInInches",
             "Census_InternalPrimaryDisplayResolutionHorizontal",
             "Census_InternalPrimaryDisplayResolutionVertical",
             "Census_GenuineStateName",
             "Census_ActivationChannel"
             ]
bool_columns=["IsBeta",
              "HasTpm",
              "IsSxsPassiveMode",
              "IsProtected",
              "Firewall",
              "UacLuaenable",
              "Census_HasOpticalDiskDrive",
              "Census_IsPortableOperatingSystem",
              "Census_IsFlightingInternal",
              "Census_IsFlightsDisabled",
              "Census_ThresholdOptIn",
              "Census_IsSecureBootEnabled",
              "Census_IsWIMBootEnabled",
              "Census_IsVirtualDevice",
              "Census_IsTouchEnabled",
              "Census_IsPenCapable",
              "Census_IsAlwaysOnAlwaysConnectedCapable",
              "Wdft_IsGamer"]
Version_columns=["EngineVersion",
                 "AppVersion",
                 "AvSigVersion",
                 "Census_OSVersion"]
data[bool_columns].fillna(2,inplace=True)
#data[num_columns].fillna(99,inplace=True)

for f in Version_columns:
    order_label = data.groupby([f])['MachineIdentifier'].count()
    data[f] = data[f].map(order_label)


df_train = data[data.HasDetections <100].copy().reset_index()
df_test = data[data.HasDetections ==100].copy().reset_index()
del df_train["index"],df_test["index"],data
gc.collect()



print(df_train.shape,df_test.shape)
features=id_columns+cate_columns+num_columns+bool_columns+Version_columns
print(len(features))
target=df_train["HasDetections"]

# def reduce_memory(df, col):
#     mx = df[col].max()
#     if mx < 256:
#         df[col] = df[col].astype('uint8')
#     elif mx < 65536:
#         df[col] = df[col].astype('uint16')
#     else:
#         df[col] = df[col].astype('uint32')
# print('Reducing memory...')
# for col in bool_columns: reduce_memory(df_train, col)
# for col in bool_columns: reduce_memory(df_test, col)

param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.75,
         "bagging_freq": 5,
         "bagging_fraction": 0.75 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "random_state": 133,
         'num_threads':16,
         "verbosity": -1}

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1993)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx],)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx],)

    num_round = 50000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                    early_stopping_rounds=200)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    current_pred= clf.predict(df_test[features], num_iteration=clf.best_iteration)

    predictions += current_pred/5
r=metrics.roc_auc_score(target, oof)
print(r)

cols = (feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",y="feature",data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('a_a_tsf_.png')

sub_df = pd.DataFrame({"MachineIdentifier": df_test["MachineIdentifier"].values})
sub_df["HasDetections"] = predictions
sub_df.to_csv("stacking/submit_"+str(r)+"_test.csv", index=False)
