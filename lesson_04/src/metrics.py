"""
Metrics

"""
import numpy as np


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)

def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()

def ap_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)    
    flags = np.isin(recommended_list, bought_list)   
    if sum(flags) == 0:
        return 0    
    sum_ = 0
    for i in range(k):       
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k            
    result = sum_ / k   
    return result

def mrr_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    flags = np.isin(recommended_list, bought_list) * 1 
    ranks = 1 / (np.array(np.where(flags == 1)[0]) + 1)
    mrr_k = ranks.sum() / len(ranks)
    
    return mrr_k

def ndcg_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    flags = np.isin(recommended_list, bought_list) * 1 
    idxs = (np.array(np.where(flags == 1)[0]) + 1)
    dcg = (1 / (np.log2(idx + 1))).sum()
    
    ideal_idxs = np.arange(len(idxs)) + 1 
    idcg = (1 / (np.log2(ideal_idxs + 1))).sum()
    
    ndcg = dcg / idcg
    return ndcg

