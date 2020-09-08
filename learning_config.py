

lconfig = {
    'epoch': 50,
    'batch_sz': 16,
    'meta_lr': 0.0005,
    'wise_lr': 0.005,

    # Semantic-wise adaptation
    's_upd_num': 1,
    # Task-wise adaptation
    't_upd_num': 1,

    # base-learner config
    # lp2r
    'layer1_dim': 64,
    'layer2_dim': 64,
    # lp2eu
    'middle_dim': 32,

    'rtx_on': True,

    #####
    'alpha': 0.5,
    'tao': 0.01,
    'beta': 0.05,
    'n_k': 3,
}

states = ["meta_training","warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]
