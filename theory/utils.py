import torch
import numpy as np


def merge_in_dict(parent, child):
    # merge child dict into parent
    for key, value in child.items():
        if key in parent:
            parent[key].append(value)
        else:
            parent[key] = [value]
    return parent

def compute_cos_sim(list_a, list_b):
    all_cos_sim = []
    eps = 1e-16
    for dfa, auto in zip(list_a, list_b):
        if (dfa is None) or (auto is None):
            all_cos_sim.append(0)
        else:
            norm = torch.norm(dfa.flatten()) * torch.norm(auto.flatten())
            if norm > 0:
                cos_sim = torch.dot(dfa.flatten(), auto.flatten())/norm
                all_cos_sim.append(cos_sim.item())
    all_cos_sim = np.array(all_cos_sim) if len(all_cos_sim) > 0 else np.array([np.nan])
    return all_cos_sim.mean().item(), all_cos_sim.std().item()


def compute_mse(list_a, list_b):
    all_mse = []
    eps = 1e-16
    for dfa, auto in zip(list_a, list_b):
        if (dfa is None) or (auto is None):
            all_mse.append(0)
        else:
            mse = torch.sum((dfa - auto)**2)//torch.sum(auto**2 + eps)
            all_mse.append(mse.item())
    all_mse = np.array(all_mse) if len(all_mse) > 0 else np.array([np.nan])
    return all_mse.mean().item(), all_mse.std().item()


def cos_sim(a, b):
    eps = 1e-10
    score = torch.dot(a.flatten(), b.flatten())/(torch.norm(a.flatten()+eps) * torch.norm(b.flatten()))
    return score.item()


class MultiplyBatchSampler(torch.utils.data.sampler.BatchSampler):
    MULTILPLIER = 2

    def __iter__(self):

        for batch in super().__iter__():
            # len(batch) = batch_size
            # list: batch
            # [1, 2, 3] * 2 = [1, 2, 3, 1, 2, 3]
            # return 1 `yield` every time __next__() is called 
            yield batch * self.MULTILPLIER