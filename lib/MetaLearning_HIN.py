

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
from lib.BaseLearning import lp2eu, lp2r
from lib.Evaluation import Evaluation
from lib.Config import config_ml
from lib.Mem import feaMem

class ml_hin(torch.nn.Module):
    def __init__(self, learning_config, data_config):
        super(ml_hin, self).__init__()

        # config
        self.lcf = learning_config
        self.dcf = data_config
        self.device = "cuda" if self.lcf['rtx_on'] else "cpu"
        self.lp2r_vars = ['w1','b1','w2','b2','w3','b3']

        # 准备base_emb_func
        if True:
            from lib.EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML
            self.item_emb = ItemEmbeddingML(config_ml)
            self.user_emb = UserEmbeddingML(config_ml)
        
        # 准备子学习机  --need args
        self.lp2eu = lp2eu(self.dcf['user_emb_dim'], self.dcf['item_emb_dim'])
        self.lp2r = lp2r(self.lcf, self.dcf['item_emb_dim'], self.dcf['user_emb_dim'])
        if(self.lcf['rtx_on']):
            self.lp2eu.cuda()
            self.lp2r.cuda()

        # 准备记忆机
        # self.mem = feaMem(self.lcf['n_k'],self.dcf['user_emb_dim'],self.lp2eu, self.device)

        # 是训练或者预测
        self.mode = 'eval'

        # 为了评估效果
        self.cal_metrics = Evaluation()
        
        # 智能优化机
        self.meta_optim = torch.optim.Adam(\
            self.parameters(), lr=self.lcf['meta_lr'])

        # 几种mp的权重 # 平均
        self.ap = {}
        


    #                       这里:train和eval不同
    # 1 task train T_u -> thita^p_u -> thita_u
    # return thita
    def forward(self, x_spt, mps_x_spt, y_spt, x_qry, mps_x_qry, y_qry, u_init): # 即mp_upd
        '''
        _g_f_v4eu = copy.copy(self.lp2eu.parameters())
        # 这里使用的profile非严格意义上的profile
        p_u = self.lp2eu(mps_x_spt['um'], self.dcf['user_emb_dim'], new_vars = _g_f_v4eu)
        bias_term, _ = self.mem.read_head(p_u, self.lcf['alpha'],\
            True if self.mode == 'train' else False)
        _g_f_v4eu['lp2eu_b'] : _g_f_v4eu['lp2eu_b'] - self.lcf['tao']*bias_term[0]
        
        global_grad = []
        '''
        # 记录各种meta path学到的参数
        X_u_s = []
        v4r_s = []
        loss_s = []

        #print('-------------one task start----------')

        for p in self.dcf['mp']:
            mp_x_spt = mps_x_spt[p]  # item near user
            mp_x_qry = mps_x_qry[p]

            # 学习参数来增强emb_user
            # fast_v4eu = self.forward_4p2eu(x_spt, mp_x_spt, y_spt, _g_f_v4eu, global_grad) # for self.mem
            fast_v4eu = self.forward_4p2eu(x_spt, mp_x_spt, y_spt)
            # 增强
            X_u = self.lp2eu(mp_x_spt, self.dcf['user_emb_dim'], new_vars = fast_v4eu)
            X_u_s.append(X_u)

            # 学习参数来rate
            fast_v4r = self.forward_4p2r(X_u, x_spt, y_spt)
            v4r_s.append(fast_v4r)

            # 计算 p类型路径 对应的 loss // 用qry集
            if(self.mode == 'train'):
                y_qry_pred = self.lp2r(X_u, x_qry, new_vars = fast_v4r)
                loss = F.mse_loss(y_qry_pred, y_qry).data
                loss_s.append(loss)

                if(p in self.ap): # 要求qry的大小要一样
                    self.ap[p] = (self.ap[p]+loss)*(0.5)
                else:
                    self.ap[p] = loss
            # test模式时不能用qry, 则取平均
            else:
                loss_s.append(self.ap[p])
        
        # self.mem.write_head(np.mean(global_grad, axis=0), self.lcf['beta'])

        # 整合几种元路径的参数
        final_X_u, final_v4r = self.aggregate(X_u_s, v4r_s, loss_s)

        # 预测
        y_qry_pred = self.lp2r(final_X_u, x_qry, new_vars = final_v4r)

        # 返回评估值 # MAML架构源代码的forward即是返回评估值
        return self.calc_eval(y_qry_pred, y_qry)

    def calc_eval(self, y_pred, y_real):

        loss = F.mse_loss(y_pred, y_real)
        y_pred = y_pred.data.cpu().numpy()
        y_real = y_real.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(y_pred, y_real)
        ndcg_5 = self.cal_metrics.ranking(y_real, y_pred, k=5)
        return loss, mae, rmse, ndcg_5


    # learn the global params
    def train_by_batch(self, x_spt_batch, y_spt_batch, mps_x_spt_batch,\
                             x_qry_batch, y_qry_batch, mps_x_qry_batch):
        self.mode = 'train'

        loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_5_s = []

        # 一个一个来
        for i in range(self.lcf['batch_sz']):

            # 先embedding来适配原论文的初始数据
            x_spt = x_spt_batch[i].to(self.device)
            x_qry = x_qry_batch[i].to(self.device)
            x_spt = self.item_emb(x_spt[:, :self.dcf['item_fea_len']])
            user_init_emb = None#self.user_emb(x_spt[:, self.dcf['item_fea_len']:])
            x_qry = self.item_emb(x_qry[:, :self.dcf['item_fea_len']])

            support_mps = dict(mps_x_spt_batch[i])  # must be dict!!!
            query_mps = dict(mps_x_qry_batch[i])    # 因为dict比list方便很多

            mps_x_spt = {}
            mps_x_qry = {}
            for p in self.dcf['mp']:
                mps_x_spt[p] = self.item_emb(torch.cat(
                    list(support_mps[p])
                ).to(self.device))
                mps_x_qry[p] = self.item_emb(torch.cat(
                    list(query_mps[p])
                ).to(self.device))

            y_spt = y_spt_batch[i].to(self.device)
            y_qry = y_qry_batch[i].to(self.device)

            # work-----------------------------------
            task_loss, task_mae, task_rmse, task_na5 = self.forward(
                x_spt, mps_x_spt, y_spt, 
                x_qry, mps_x_qry, y_qry,
                user_init_emb
            )

            loss_s.append(task_loss)
            mae_s.append(task_mae)
            rmse_s.append(task_rmse)
            ndcg_at_5_s.append(task_na5)

        # 反向传播
        loss = torch.stack(loss_s).mean(0)
        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()

        return loss.cpu().data.numpy(), np.mean(mae_s), np.mean(rmse_s), np.mean(ndcg_at_5_s)

    def forward_4p2eu(self, emb_item, emb_item_near_user, y_spt, global_f_v4eu = None, global_grad = None):
        
        if global_f_v4eu == None:
            fast_v4eu = copy.copy(self.lp2eu.parameters())
        else:
            fast_v4eu = global_f_v4eu
        
        # updates
        for i in range(self.lcf['s_upd_num']):
            X_u = self.lp2eu(emb_item_near_user, self.dcf['user_emb_dim'], new_vars = fast_v4eu)
            y_spt_pred = self.lp2r(X_u, emb_item)
            #print(fast_v4eu['lp2eu_b'], fast_v4eu['lp2eu_w'])
            loss = F.mse_loss(y_spt_pred, y_spt)
            grad = torch.autograd.grad(loss, fast_v4eu.values(), \
                create_graph=True) # graph?
            
            if global_grad != None:
                global_grad.append(copy.deepcopy(grad[1].data).cpu().numpy()) # for self.mem

            fast_v4eu = {
                'lp2eu_w': fast_v4eu['lp2eu_w'] - self.lcf['wise_lr']*grad[0],
                'lp2eu_b': fast_v4eu['lp2eu_b'] - self.lcf['wise_lr']*grad[1]
            }
            
        return fast_v4eu

    def forward_4p2r(self, X_u, emb_item, y_spt):

        global_v4r = self.lp2r.parameters()
        
        # 把参数w放到 p型元路径 的有效空间里，即 w->w^p
        fast_v4r = self.Transformation_Function(global_v4r, X_u)
        # 然后再学习
        for i in range(self.lcf['t_upd_num']):
            y_spt_pred = self.lp2r(X_u, emb_item, new_vars = fast_v4r)

            loss = F.mse_loss(y_spt_pred, y_spt)
            grad = torch.autograd.grad(loss, fast_v4r.values(), \
                create_graph=True) # graph?
            for i,k in enumerate(fast_v4r):
                fast_v4r[k] = fast_v4r[k] - self.lcf['wise_lr']*grad[i]

        return fast_v4r

    def Transformation_Function(self, global_v4r, X_u): # X^p<S>_u

        fast_v4r = {}
        for k in self.lp2r_vars:
            fast_v4r[k] = global_v4r[k] *\
                torch.sigmoid(
                    nn.Linear(  # [x, in] -> [x, out]
                        self.dcf['user_emb_dim'], # in_feature
                        np.prod(global_v4r[k].shape), # out_feature
                    ).cuda().forward(
                        X_u
                    )
                ).view(global_v4r[k].shape)

        return fast_v4r

    def aggregate(self, X_u_s, v2r_s, loss_s):

        att = F.softmax(-torch.stack(loss_s), dim = 0)

        final_v2r = {}
        for k in self.lp2r_vars:
            final_v2r[k] = 0
            for i in range(len(self.dcf['mp'])):
                final_v2r[k] += (att[i] * v2r_s[i][k])

        return torch.sum(torch.stack(X_u_s) * att.unsqueeze(1), 0), final_v2r

    # return task loss 做实验嘛，评估而不是做预测/和train等价了
    def evaluate(self, _x_spt, _y_spt, _mps_x_spt, _x_qry, _y_qry, _mps_x_qry):
        self.mode = 'eval'

        x_spt = _x_spt.to(self.device)
        x_qry = _x_qry.to(self.device)
        x_spt = self.item_emb(x_spt[:, :self.dcf['item_fea_len']])
        x_qry = self.item_emb(x_qry[:, :self.dcf['item_fea_len']])

        support_mps = dict(_mps_x_spt)  # must be dict!!!
        query_mps = dict(_mps_x_qry)

        mps_x_spt = {}
        mps_x_qry = {}
        for p in self.dcf['mp']:
            mps_x_spt[p] = self.item_emb(torch.cat(
                list(support_mps[p])
            ).to(self.device))
            mps_x_qry[p] = self.item_emb(torch.cat(
                list(query_mps[p])
            ).to(self.device))

        y_spt = _y_spt.to(self.device)
        y_qry = _y_qry.to(self.device)

            # work-----------------------------------
        _, mae, rmse, ndcg_5 = self.forward(
            x_spt, mps_x_spt, y_spt, 
            x_qry, mps_x_qry, y_qry,
            1
        )

        return mae, rmse, ndcg_5
