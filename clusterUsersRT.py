#!/usr/bin/env python
# coding: utf-8

###############################################################################
#   Code written by Peter Stefanov (Sofia University)
#   p.stefanov@hotmail.com, stefanov.peter.ps@gmail.com
#   and modified by Kareem Darwish (Qatar Computing Research Institute)
#   kdarwish@hbku.edu.qa
#   The code is provided for research purposes ONLY
###############################################################################

###############################################################################
# The script takes two arguments
# sys.argv[1] is a tab separated file with first column containing UserIDs
# and second column containing tweets
# sys.argv[2] is the output file with userID,x-coordinates,y-coordinates,clusterID
# cluser ID of -1 means the item was not clustered.
###############################################################################

import argparse
import re

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MeanShift

###############################################################################
# Funcs
###############################################################################

def extract_rt(text):
    try:
        m = re.match(r'^RT:? +@(.*?):', text)
        if m:
            return m.group(1).lower()
        return np.nan
    except:
        return np.nan


class Enumerator:
    def __init__(self):
        self.i = 0
        self.dict = dict()

    def __getitem__(self, item):
        if item not in self.dict:
            save = self.i
            self.dict[item] = self.i
            self.i += 1
            return save
        return self.dict[item]

    def __contains__(self, value):
        return value in self.dict

###############################################################################
# Main
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('source_file', help="csv file with 2 columns: user and Tweet; no header, standard comma-sep")
parser.add_argument('output_file', help="csv file with 1st column user, followed by coord. columns, followed by cluster_id; no header, standard comma-sep")
args = parser.parse_args()

df_text = pd.read_csv(args.source_file, header=None, usecols=[0, 1], sep='\t', error_bad_lines=False)
df_text.columns = ['User', 'Text']
df_text = df_text.apply(lambda s: s.str.strip())
df_text.loc[:, 'User'] = df_text.User.str.lower()

# Extract retweeted accounts and work only with those rows
df_text['Retweet'] = df_text.Text.apply(extract_rt)
df_text.dropna(subset=['Retweet'], inplace=True)

print(f"Number of Retweets: {len(df_text)}")
print(f"Number of unique users after filtering only Retweets: {df_text.User.nunique()}")

# Sample users
min_nunique_retweet = 5
sample_size = 5000

# take `sample_size` most active
users_of_interest = df_text.groupby('User')['Retweet'].nunique() \
    .where(lambda x: x >= min_nunique_retweet).dropna() \
    .sort_values(ascending=False).head(n=sample_size)

# Filter df_text to have only the Tweets of the sampled users
df_text = df_text[df_text.User.isin(users_of_interest.index)]

# Calculate similarity
user_feature_counts = df_text.groupby(['User', 'Retweet']).size()
user2idx = Enumerator()
feature2idx = Enumerator()

row_ind = []
col_ind = []
data = []
for (user, feature), count in user_feature_counts.items():
    row_i = user2idx[user]
    col_i = feature2idx[feature]
    row_ind.append(row_i)
    col_ind.append(col_i)
    data.append(count)

user_feature_matrix = sparse.csr_matrix((data, (row_ind, col_ind)))
user2user_sim = cosine_similarity(user_feature_matrix).clip(max=1.0) # clip, because sometimes get values like 1.0000000014

# Dimentionality reduction
user_points = UMAP(metric='precomputed').fit_transform(1 - user2user_sim) # works with distances, NOT similarity

# Scale user vectors between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
user_points_scaled = scaler.fit_transform(user_points)

idx2user = {v:k for k, v in user2idx.dict.items()}
df_user = pd.DataFrame(user_points_scaled, index=map(idx2user.get, range(len(user_points_scaled))))

# Clustering
clusters = MeanShift(cluster_all=False, bin_seeding=True).fit(df_user.values)
df_user['cluster_id'] = clusters.labels_

# Regard users in very small clusters as unclustered
for c in df_user['cluster_id'].unique():
    if c == -1:
        continue
    if len(df_user[df_user['cluster_id'] == c]) < len(df_user[df_user['cluster_id'] != -1]) * 0.01:
        df_user.loc[df_user['cluster_id'] == c, 'cluster_id'] = -1

# Generate output
df_user[df_user.cluster_id != -1].to_csv(args.output_file, header=False)
