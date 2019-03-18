import numpy as np
#类别变量处理  treat categorical features as numerical ones
#1 labelencode
#2 frequency encoding
#3  mean-target encoding
#https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/79981
# Mean-target encoding is a popular technique to treat categorical features as numerical ones.
# The mean-target encoded value of one category is equal to the mean target of all samples of
# the corresponding category (plus some optional noise for regularization).


from sklearn.preprocessing import LabelEncoder

for name in f_feature:
    lab = LabelEncoder()
    data["le_" + name] = lab.fit_transform(data[name].astype("str"))

data["%s_count_num"%fea]=data[fea].map(data[fea].value_counts(dropna=False))


def frequency_encoding(variable):
    t = df_train[variable].value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']

for variable in ['age', 'network_age']:
    freq_enc_dict = frequency_encoding(variable)
    data[variable+"_freq"] = data[variable].map(lambda x: freq_enc_dict.get(x, np.nan))
