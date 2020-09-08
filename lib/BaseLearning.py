


import torch
from torch import nn
from torch.nn import functional as F
import copy

# lp2eu: learn_param_to_emb_user
class lp2eu(nn.Module):
    def __init__(self, user_emb_dim, item_emb_dim):
        super(lp2eu, self).__init__()
        
        self.vars = nn.ParameterDict() 

        # params for the linear
        # w
        w = nn.Parameter(\
                    torch.ones([user_emb_dim, item_emb_dim]))  
        nn.init.xavier_normal_(w)
        self.vars['lp2eu_w'] = w
        # b
        self.vars['lp2eu_b'] = nn.Parameter(torch.zeros(user_emb_dim))
    
    def forward(self, emb_item_near_user, user_emb_dim, new_vars = None):
        # update循环时会用到新的vars
        new_vars = self.vars if new_vars == None else new_vars

        x = F.leaky_relu(
            torch.mean( #mean pooling
                F.linear(
                    emb_item_near_user, 
                    new_vars['lp2eu_w'], 
                    new_vars['lp2eu_b']
                ), 0
            )
        )

        return x 

    def parameters(self):
        return self.vars

    def zero_grad(self):  
        with torch.no_grad():
            for v in self.vars.values():
                if v.grad is not None:
                    v.grad.zero_()

    def get_weights(self):
        return copy.deepcopy(self.vars['lp2eu_b'])

    def get_zero_weights(self):
        return copy.deepcopy(torch.zeros_like(self.vars['lp2eu_b']))


# lp2r: learn_param_to_rate
# g_emb_user: X_u (calc/featured by emb_item)
class lp2r(nn.Module):
    def __init__(self, lconfig, item_emb_dim, user_emb_dim):
        super(lp2r, self).__init__()
        
        self.vars = nn.ParameterDict() 

        # layer-1
        w1 = nn.Parameter(\
            torch.ones([
                lconfig['layer1_dim'], 
                item_emb_dim + user_emb_dim
            ]))  
        nn.init.xavier_normal_(w1)  # Xavier方法的权值初始化
        self.vars['w1'] = w1
        self.vars['b1'] = nn.Parameter(torch.zeros(lconfig['layer1_dim'])) # Parameter方法把参数放进model里，于是之后可以更新

        # layer-2
        w2 = nn.Parameter(\
            torch.ones([
                lconfig['layer2_dim'],
                lconfig['layer1_dim']
            ]))
        nn.init.xavier_normal_(w2)
        self.vars['w2'] = w2                                          # 嗯? 和原代码不同?
        self.vars['b2'] = nn.Parameter(torch.zeros(lconfig['layer2_dim']))

        # output linear
        w3 = nn.Parameter(torch.ones([1, lconfig['layer2_dim']]))
        nn.init.xavier_normal_(w3)
        self.vars['w3'] = w3
        self.vars['b3'] = nn.Parameter(torch.zeros(1))

    def forward(self, X_u, emb_item, new_vars=None):
        # update循环时会用到新的vars
        new_vars = self.vars if new_vars == None else new_vars
        
        x = torch.cat((
                emb_item, 
                X_u.repeat(emb_item.shape[0], 1)  # cat需要维度适配
            ), 1) 

        # 2-lyer mlp
        x = F.relu(\
            F.linear(x, \
                new_vars['w1'], new_vars['b1']))
        x = F.relu(\
            F.linear(x, \
                new_vars['w2'], new_vars['b2']))
        # output
        x = F.linear(x, \
                new_vars['w3'], new_vars['b3'])

        return x.squeeze()  # F.linear之后是会参杂1-维度的

    def parameters(self):
        return self.vars

    def zero_grad(self):  
        with torch.no_grad():
            for v in self.vars.values():
                if v.grad is not None:
                    v.grad.zero_()