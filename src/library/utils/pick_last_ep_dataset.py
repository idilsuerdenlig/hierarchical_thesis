import numpy as np


def pick_last_ep(dataset):

    ep_end_indices = list()
    for i, dataset_step in enumerate(dataset):
        if dataset_step[-1]:
            ep_end_indices.append(i)

    last_ep_begin = ep_end_indices[-2]
    last_ep_end = ep_end_indices[-1]
    return dataset[last_ep_begin:last_ep_end+1]
