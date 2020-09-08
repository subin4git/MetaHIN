
import torch

class feaMem: # 记录用户与其init_weight的对应
    def __init__(self, n_k, u_emb_dim, base_model, device):
        self.n_k = n_k
        self.base_model = base_model
        self.p_memory = torch.randn(n_k, u_emb_dim, device=device).normal_()        
        u_param = base_model.get_weights()  ###########
        self.u_memory = []
        for i in range(n_k):
            bias_list = []
            for param in u_param:
                bias_list.append(param.normal_(std=0.05))
            self.u_memory.append(bias_list)
        self.att_values = torch.zeros(n_k).to(device)
        self.device = device

    def read_head(self, p_u, alpha, train=True):
        # a_u
        att_model = Attention(self.n_k).to(self.device)
        attention_values = att_model(p_u, self.p_memory).to(self.device)  # pu on device
        # get personalized mu
        personalized_mu = self.base_model.get_zero_weights()  #####
        #print(attention_values)
        #print(torch.tensor(self.u_memory).shape)
        _att_values = attention_values#.reshape(len(self.u_memory),1)
        #print(_att_values)
        for i in range(len(self.u_memory)):
            for j in range(len(self.u_memory[i])):
                #print(i, j, self.u_memory[i][j])
                #print(_att_values[i])
                #print(personalized_mu[j])
                personalized_mu[j] += _att_values[i] * self.u_memory[i][j].to(self.device)
        # update mp
        transposed_att = attention_values.reshape(self.n_k, 1)
        product = torch.mm(transposed_att, p_u.unsqueeze(0))
        if train:
            self.p_memory = alpha * product + (1-alpha) * self.p_memory
        self.att_values = attention_values
        return personalized_mu, attention_values

    def write_head(self, u_grads, lr):
        att_values = self.att_values.reshape(len(self.u_memory), 1)
        for i in range(len(self.u_memory)):
            for j in range(len(self.u_memory[i])):
                self.u_memory[i][j] = (lr * att_values[i] * u_grads[j] + (1-lr) * self.u_memory[i][j])[0]



def cosine_similarity(input1, input2):
    query_norm = torch.sqrt(torch.sum(input1**2+0.00001, 1))
    doc_norm = torch.sqrt(torch.sum(input2**2+0.00001, 1))

    prod = torch.sum(torch.mul(input1, input2), 1)
    norm_prod = torch.mul(query_norm, doc_norm)

    cos_sim_raw = torch.div(prod, norm_prod)
    return cos_sim_raw  

class Attention(torch.nn.Module):
    def __init__(self, n_k):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k, torch.nn.LeakyReLU())
        self.soft_max_layer = torch.nn.Softmax(dim=0)

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        inputs = cosine_similarity(expanded_pu, mp)
        fc_layers = self.fc_layer(inputs)
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values