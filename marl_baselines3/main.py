
import config
import time
from multiprocessing import Process
import argparse
import os
import copy

import shutil
import torch as th
import random
import numpy as np

import random
import numpy as np
import torch.nn as nn
from typing import Tuple

import math
import torch.nn.functional as F
import numpy as np
seed = 42
np.random.seed(seed)
th.manual_seed(seed)
random.seed(seed)

if th.cuda.is_available():
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
th.use_deterministic_algorithms(True)
from cityflow_env import CityFlowEnv
from independent_ppo import PPO
from evaluate_policy import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy

def lane_relation(phase_map):
    num_lanes=24
    lane_relation = np.zeros((num_lanes, num_lanes))

    for phase_lanes in phase_map:
        for i in phase_lanes:
            for j in phase_lanes:
                
                lane_relation[i, j] = 1
    return lane_relation
  
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result
phase_map=[[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
relation=lane_relation(phase_map)

class MyMultiHeadAttention(nn.Module):

    def __init__(self, d_model=128, num_heads=4):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.bias=nn.Parameter(th.zeros(num_heads))
        self.bias2=nn.Parameter(th.zeros(num_heads))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        B, N, D = x.shape

        # --------------------------------
        # Q, K, V
        # --------------------------------

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # [B, N, D]
        #      ↓
        # [B, H, N, head_dim]

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # [B, H, N, head_dim]

        # --------------------------------
        # Attention scores
        # --------------------------------

        scores = th.matmul(
            Q,
            K.transpose(-2, -1)
        )
        phase_map=[[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
        relation=lane_relation(phase_map)
        
        relation = th.as_tensor(relation,dtype=scores.dtype,device=scores.device)
        relation=relation.repeat_interleave(10, dim=0).repeat_interleave(10, dim=1)
        I=th.eye(24)
        I=I.repeat_interleave(10, dim=0).repeat_interleave(10, dim=1)
        I = th.as_tensor(I,dtype=scores.dtype,device=scores.device)
        scores = scores / (self.head_dim ** 0.5) + self.bias[None, :, None, None]*relation[None , None , : , :]+self.bias2[None, :, None, None]*I[None , None , : , :]

        # [B, H, N, N]

        # --------------------------------
        # Mask
        # --------------------------------

        if mask is not None:
            scores = scores.masked_fill(
                mask[:, None, None, :],
                float("-inf")
            )

        # --------------------------------
        # Softmax
        # --------------------------------

        attention = th.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # Weighted sum
        # --------------------------------

        output = th.matmul(
            attention,
            V
        )

        # [B, H, N, head_dim]

        # --------------------------------
        # Combine heads
        # --------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            B, N, self.d_model
        )

        output = self.out_proj(output)

        return output, attention
class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MyMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Attention
        attn_out, attention = self.attention(
            x,
            mask
        )

        x = self.norm1(
            x + self.dropout(attn_out)
        )

        # Feed-forward
        ff_out = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ff_out)
        )

        return x, attention
class SelfAttention(nn.Module):
    def __init__(self, d, d_q, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.latent_dim_pi = 4
        self.latent_dim_vf = 4

        self.transformerblock1=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)
        self.transformerblock2=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)
        self.transformerblock3=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)
        self.transformerblock4=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,dim_feedforward=64, nhead=4, dropout=0, batch_first=True),
            num_layers=2
        )


        self.transformer2=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,dim_feedforward=64, nhead=4, dropout=0, batch_first=True),
            num_layers=2
        )
        self.self_attn1 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )

        self.linear1 = nn.Linear(2, 32)
        self.linear2 = nn.Linear(2, 32)
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


        
        #context_vector2=self.policy_net2(context_vector2)
        return self.forward_actor(x),self.forward_critic(x)


    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        x[: , : ,39]=0
        mask=x[: , : , 30:].reshape(-1 , 240 ).bool()
        x=x[: , : , :20].reshape(-1 , 240 , 2) 
        x=self.linear1(x)
    
        x , _=self.transformerblock1(x , mask)
        x, _=self.transformerblock2(x , mask)        
        

        x=x.reshape(-1, 24, 10 , 32)
        mask=mask.reshape(-1, 24, 10 ).unsqueeze(-1)
        phase_feats_map_1=[]
        for i in range(4):
          
          tmp_feat_1 = x[:, self.phase_map[i], : , :]*(~mask[:, self.phase_map[i], : , :])
          tmp_feat_1_mean = tmp_feat_1.mean(dim=(1, 2),keepdim=True).squeeze(1)

          phase_feats_map_1.append(tmp_feat_1_mean)
             
        phase_feat_all = th.cat(phase_feats_map_1, dim=1)


        hidden=self.linear3(phase_feat_all)
        hidden=self.Relu1(hidden) 
        hidden=self.linear5(hidden) 
        hidden=self.Relu3(hidden)        
        hidden=self.linear7(hidden)   
     

        return hidden.reshape(-1 , 4)


    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        x[: , : ,39]=0
        mask=x[: , : , 30:].reshape(-1 , 240 ).bool()
        x=x[: , : , :20].reshape(-1 , 240 , 2)
        x=self.linear2(x)
        x , _=self.transformerblock3(x , mask)
        x , _=self.transformerblock4(x , mask)

        x=x.reshape(-1, 24, 10 , 32)
        mask=mask.reshape(-1, 24, 10 ).unsqueeze(-1)

        phase_feats_map_1=[]
        for i in range(4):
          tmp_feat_1 = x[:, self.phase_map[i], : , :]*(~mask[:, self.phase_map[i], : , :])
          tmp_feat_1_mean = tmp_feat_1.mean(dim=(1, 2),keepdim=True).squeeze(1)
          phase_feats_map_1.append(tmp_feat_1_mean)
              
        phase_feat_all = th.cat(phase_feats_map_1, dim=1)
        hidden=self.linear4(phase_feat_all)  
        hidden=self.Relu2(hidden)
        hidden=self.linear6(hidden)
        hidden=self.Relu4(hidden)
        hidden=self.linear8(hidden) 
        
        return hidden.reshape(-1 , 4)




# Custom Policy incorporating the Self-Attention feature extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):

        use_sde = kwargs.pop('use_sde', False)
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,


            *args,
            use_sde=use_sde,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:

        # Use the shared extracted features for both policy and value networks
        self.features_extractor = nn.Identity()
        self.mlp_extractor = SelfAttention(8,8,8,8)
        self.pi_features_extractor= nn.Identity()
        self.vf_features_extractor=nn.Identity()
        self.action_net=nn.Identity()
        self.value_net=nn.Identity()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='AttendLight')
    parser.add_argument("--mod", type=str, default="Attend")
    parser.add_argument("--model", type=str, default="AttendLight")
    parser.add_argument("--proj_name", type=str, default="chatgpt-TSCS-Transfer")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--multi_process", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hangzhou", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--duration", type=int, default=30)
    return parser.parse_args()


def main(in_args=None):
    traffic_file_list = []

    if in_args.dataset == 'jinan':
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24000_60min.json"]
        num_rounds = 100
        template = "Jinan"
    elif in_args.dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json", "anon_4_4_hangzhou_synthetic_24000_60min.json"]
        num_rounds = 100
        template = "Hangzhou"
    elif in_args.dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        num_rounds = 100
        template = "NewYork"

    # flow_file error
    try:
        if in_args.traffic_file not in traffic_file_list:
            raise error.flowFileException('Flow file does not exist.')
    except error.flowFileException as e:
        print(e)
        return
    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)
    process_list = []
    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": num_rounds,
        "NUM_GENERATORS": in_args.gen,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_intersections,
        "RUN_COUNTS": count,

        "MODEL_NAME": in_args.mod,
        "MODEL": in_args.model,
        "PROJECT_NAME": in_args.proj_name,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
        "TRAFFIC_SEPARATE": in_args.traffic_file,
        "LIST_STATE_FEATURE": [
            "num_in_seg_attend",
        ],

        "DIC_REWARD_INFO": {
            "pressure": -0.25,
        },
    }

    if in_args.eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                    'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", in_args.memo, in_args.traffic_file + "_"
                                      + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_"
                                               + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
        "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
    }
    

    os.makedirs(dic_path_extra["PATH_TO_WORK_DIRECTORY"], exist_ok=True)
    config.dic_traffic_env_conf['MIN_ACTION_TIME'] = in_args.duration
    config.dic_traffic_env_conf['MEASURE_TIME'] = in_args.duration
    deploy_dic_agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)
    path = deploy_dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(deploy_dic_path["PATH_TO_DATA"], deploy_dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], deploy_dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(deploy_dic_path["PATH_TO_DATA"], deploy_dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], deploy_dic_traffic_env_conf["ROADNET_FILE"]))
    
    env=CityFlowEnv(deploy_dic_path["PATH_TO_WORK_DIRECTORY"],deploy_dic_path["PATH_TO_WORK_DIRECTORY"], deploy_dic_traffic_env_conf , deploy_dic_path)

    return env

if __name__ == "__main__":
    args = parse_args()
    datasets={"jinan":["anon_3_4_jinan_real.json" , "anon_3_4_jinan_real_2000.json" , "anon_3_4_jinan_real_2500.json" , "anon_3_4_jinan_synthetic_24000_60min.json"], "hangzou":["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json", "anon_4_4_hangzhou_synthetic_24000_60min.json"], "new_york":["anon_28_7_newyork_real_double.json" , "anon_28_7_newyork_real_triple.json"]}
    env = main(args)
    ppo=PPO(CustomPolicy  ,  env , verbose=1 ,batch_size=30, n_steps=120)
    ppo.learn(env, total_timesteps=24000 , log_interval=1)
        
    for dataset, traffic_files in datasets.items():
        for traffic_file in traffic_files:
            args.dataset = dataset
            args.traffic_file = traffic_file
            env = main(args)
            mean_reward, std_reward, mean_queue_length, std_queue_length, mean_queue_num, std_queue_num, mean_waiting_time, std_waiting_time, mean_travel_time, std_travel_time = evaluate_policy(ppo , env)
            print("mean_reward:", mean_reward,"std_reward", std_reward, "mean_queue_length:" , mean_queue_length, "std_queue_length:" ,std_queue_length, "mean_queue_num:", mean_queue_num, "std_queue_num:", std_queue_num, "mean_waiting_time:" ,mean_waiting_time, "std_waiting_time:" ,std_waiting_time, "mean_travel_time:" ,mean_travel_time, "std_travel_time:" ,std_travel_time )    
 
