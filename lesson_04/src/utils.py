"""
prefilter_items function,
postfilter_items function

"""
import numpy as np

def prefilter_items(data, take_n_popular=5000):
    """Отбирает наиболее популярные товары"""
    
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top_k = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    data.loc[~data['item_id'].isin(top_k), 'item_id'] = 999999
    
    return data
    
def postfilter_items(user_id, recommednations):
    pass