import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np


class presslight(nn.Module):
    def __init__(self, action_dim):
        super(presslight, self).__init__()
        self.shared_hidden = nn.Sequential(
            nn.Linear(20, 20),
            nn.Sigmoid()
        )
        self.q_networks = nn.ModuleList()
        for phase_id in range(action_dim):
            
            self.q_networks.append(
                nn.Sequential(
                    nn.Linear(20, 20),
                    nn.ReLU(),
                    nn.Linear(20, action_dim)
                )
            )
        self.phase_map= [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]]
        self.register_buffer("phase",th.tensor([[0, 1, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 1],[1, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 1, 0]], dtype=th.float32))
    def forward(self, x: th.Tensor) -> th.Tensor:
        shared=self.shared_hidden(x)
        list_selected_q_values=[]
        for index, branch in enumerate(self.q_networks):
        
            selector = th.isclose(
                x[: ,:8],
                self.phase[index].unsqueeze(0),
            ).all(dim=-1, keepdim=True)
          
            q_values=branch(shared)
            list_selected_q_values.append(q_values* selector.to(x.dtype))
        q_values=sum(list_selected_q_values)

        
        #context_vector2=self.policy_net2(context_vector2)
        return q_values




