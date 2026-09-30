
class AttendLight(nn.Module):
    def __init__(self, d, d_q, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.latent_dim_pi = 4
        self.latent_dim_vf = 4

        self.self_attn1 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )

        self.linear1 = nn.Linear(4, 32)
        self.linear2 = nn.Linear(4, 32)
        self.linear3 = nn.Linear(32, 20)
        self.linear4 = nn.Linear(32, 20)
        self.linear5 = nn.Linear(20, 20)
        self.linear6 = nn.Linear(20, 20)
        self.linear7 = nn.Linear(20, 1)
        self.linear8 = nn.Linear(20, 1)
        self.Relu1 = nn.ReLU()
        self.Relu2 = nn.ReLU()
        self.Relu3 = nn.ReLU()
        self.Relu4 = nn.ReLU()
        self.self_attn2 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        self.self_attn3 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        self.self_attn4 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        self.phase_map= [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]]


    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:

        
        return self.forward_actor(x),self.forward_critic(x)


    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        x=x.reshape(-1, 24, 4)
        x=self.linear1(x)
        phase_feats_map_1=[]
        for i in range(4):
          
          tmp_feat_1 = x[:, self.phase_map[i], :]
          tmp_feat_1_mean = tmp_feat_1.mean(dim=1,keepdim=True)
          tmp_feat_2  , _= self.self_attn1(tmp_feat_1_mean, tmp_feat_1 , tmp_feat_1)

          phase_feats_map_1.append(tmp_feat_2)
             
        phase_feat_all = th.cat(phase_feats_map_1, dim=1)

        phase_attention , _= self.self_attn3(phase_feat_all, phase_feat_all, phase_feat_all)

        hidden=self.linear3(phase_attention)
        hidden=self.Relu1(hidden) 
        hidden=self.linear5(hidden) 
        hidden=self.Relu3(hidden)        
        hidden=self.linear7(hidden)   
     

        return hidden.reshape(-1 , 4)


    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        x=x.reshape(-1, 24, 4)
        x=self.linear2(x) 
        phase_feats_map_1=[]
        for i in range(4):
          tmp_feat_1 = x[:, self.phase_map[i], :]
          tmp_feat_1_mean = tmp_feat_1.mean(dim=1,keepdim=True)
          tmp_feat_2  , _= self.self_attn2(tmp_feat_1_mean, tmp_feat_1 , tmp_feat_1)
          phase_feats_map_1.append(tmp_feat_2)
              
        phase_feat_all = th.cat(phase_feats_map_1, dim=1)
        phase_attention  , _= self.self_attn4(phase_feat_all, phase_feat_all , phase_feat_all)
        hidden=self.linear4(phase_attention)  
        hidden=self.Relu2(hidden)
        hidden=self.linear6(hidden)
        hidden=self.Relu4(hidden)
        hidden=self.linear8(hidden)                 


        

           
   
        
        return hidden.reshape(-1 , 4)

