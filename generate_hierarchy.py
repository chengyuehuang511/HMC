import pandas as pd
import numpy as np
import torch
virus_more_seq = pd.read_csv("C:/Users/v-chenghuang/Desktop/virus_more_seq_tax_date_prot_new.csv", index_col=0)
hier_name = ['realm', 'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'species']

hier_info = []
for h in hier_name:
    print(len(virus_more_seq[h].value_counts()))
    hier_info.append(virus_more_seq[h].value_counts())

h_specific = []
for i in range(2):  # len(hier_name) - 1
    L12_table = []
    for jj in range(len(hier_info[i])):
        L12_table.append([])
    for L2_label, idx in enumerate(hier_info[i + 1].index):
        info = virus_more_seq[virus_more_seq[hier_name[i + 1]] == idx][hier_name[i]].value_counts()
        assert len(info) == 1
        L12_table[np.where(hier_info[i].index == info.index[0])[0].item()].append(L2_label)
print(1)