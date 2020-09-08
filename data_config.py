

dconfig = {
    
    'dataset': 'movielens',
    # 'mp': ['um'],
    # 'mp': ['um','umdm'],
     'mp': ['um','umam','umdm'],
    # 'mp': ['um','umum','umam','umdm'],
    'file_num': 12,  # each task contains 12 files for movielens

    'item_fea_len': 26,

    'embedding_dim': 32,
    'user_emb_dim': 32*4,  # 4 features  # 也是X_u的dim(不另加fea的话)
    'item_emb_dim': 32*2,  # 2 features
}