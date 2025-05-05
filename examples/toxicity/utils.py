import json
import numpy as np
import random
import os
import pandas as pd
from datasets import load_dataset

def get_split(slice_index, pos_instances, neg_instances, template_str, user_tag="", assistant_tag="", split_type="train"):
    
    s_pos_instances = pos_instances["train"].select(slice_index)
    s_neg_instances = neg_instances["train"].select(slice_index)

    paired_data = [[pos, neg] for pos, neg in zip(s_pos_instances["prompt"], s_neg_instances["prompt"])]

    labels = []
    if split_type == "train":
        for d in paired_data:
            true_s = d[0]
            random.shuffle(d)
            labels.append([s == true_s for s in d])

    else:
        labels.extend([[1,0]]* len(paired_data))
        
    data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in np.concatenate(paired_data).tolist()]
    return  {'data': data, 'labels': labels},

    
def toxicgen_concept_dataset(n_train=150,user_tag="", assistant_tag="", seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the toxicity of the following scenario.\nScenario: {scenario}\nThe amount of toxicity in the scenario is {assistant_tag} '

    dataset = load_dataset("toxigen/toxigen-data","train").shuffle(seed=1)
    
    pos_instances = dataset.filter(lambda x: x['prompt_label'] == 1)
    neg_instances = dataset.filter(lambda x: x['prompt_label'] == 0)
    
    return {
            'train': get_split(range(0,n_train),pos_instances,neg_instances,template_str,user_tag,assistant_tag, "train")[0],
            'val': get_split(range(n_train, n_train*2),pos_instances,neg_instances,template_str,user_tag,assistant_tag,"val")[0],
            'test': get_split(range(n_train*2,n_train*5),pos_instances,neg_instances,template_str,user_tag,assistant_tag,"test")[0]
        }