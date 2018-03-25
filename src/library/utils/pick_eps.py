import numpy as np


def pick_eps(dataset, start, end):

    ep_end_indices = list()
    ep_end_indices.append(-1)
    for i, dataset_step in enumerate(dataset):
        if dataset_step[-1]:
            ep_end_indices.append(i)
    start_ep_begin = ep_end_indices[start]+1
    end_ep_end = ep_end_indices[end+1]

    return dataset[start_ep_begin:end_ep_end+1]
