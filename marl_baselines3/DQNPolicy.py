from typing import Any

import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


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


    def forward(self, x: th.Tensor) -> th.Tensor:


        
        #context_vector2=self.policy_net2(context_vector2)
        return self.forward_actor(x)


    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        x=x.reshape(-1, 24, 40)
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
        x=x.reshape(-1, 24, 40)
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

class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = SelfAttention(8 ,8, 8,8)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """

        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data





class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)




    
    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)



    
    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data



    
    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode



