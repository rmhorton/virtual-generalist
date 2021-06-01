# extract attribute names py
# Read the attributes file, and just take the keys - the attribute names.  
# JMA 23 May 2021

import os, re, sys
import json

FN = './doc/attributes.json'
attr_dict = json.load(open(FN, 'rb'))
kad = attr_dict.keys()
k_set = set()
null_write_set = set()
one_write_set = set()
multi_write_set = set()
for k in kad:
    # Find the write keys & build a set of them. 
    k_set.add(k)
    write_keys = list(attr_dict[k]['write'].keys())
    print(len(write_keys), end = ':')
    if len(write_keys)  == 0:
        null_write_set.add(k)
    elif len(write_keys)  == 1:
        one_write_set.add(write_keys[0])
    else:
        multi_write_set.add(write_keys[0])

print('\n',len(null_write_set), len(one_write_set), len(multi_write_set))
print('\nAll Attributes\n', 35*'-','\n', sorted(list(k_set)))