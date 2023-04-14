import pandas as pd
import numpy as np
import torch

"""preprocess
virus_more_seq = pd.read_csv("C:/Users/HCY/Desktop/virus_more_seq_tax_date_prot_new.csv", index_col=0, dtype=str)
virus_more_seq = virus_more_seq[~virus_more_seq['realm'].isna()]
hier_name = ['realm', 'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'species']

for h in hier_name:
    nan_idx = (virus_more_seq[h] == 'unknown') | (virus_more_seq[h] == 'environmental samples')
    virus_more_seq[h][nan_idx] = np.nan

for i in range(len(virus_more_seq)):
    for idx, h in enumerate(hier_name):
        if str(virus_more_seq.iloc[i][h]) == 'nan':
            virus_more_seq.iloc[i][hier_name[idx:]] = np.nan
            break

# virus_more_seq.to_csv("C:/Users/HCY/Desktop/virus_more_seq_nan.csv")
virus_more_seq.to_csv("C:/Users/HCY/Desktop/virus_more_seq_nan_clean.csv")
"""

virus_more_seq = pd.read_csv("C:/Users/HCY/Desktop/virus_more_seq_nan_clean.csv", index_col=0, dtype=str)
hier_name = ['realm', 'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'species']

hier_info = []
for h in hier_name:
    print(len(virus_more_seq[h].value_counts()))
    hier_info.append(virus_more_seq[h].value_counts())

h_specific = []
for i in range(len(hier_name) - 1):  # len(hier_name) - 1
    L12_table = []
    for jj in range(len(hier_info[i])):
        L12_table.append([])
    for L2_label, idx in enumerate(hier_info[i + 1].index):
        info = virus_more_seq[virus_more_seq[hier_name[i + 1]] == idx][hier_name[i]].value_counts()
        assert len(info) == 1
        # print(info, idx)
        L12_table[np.where(hier_info[i].index == info.index[0])[0].item()].append(L2_label)
    h_specific.append(L12_table)
print(1)
