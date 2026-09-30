import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np

def relation(phase_list):
    relations = []
    num_phase = len(phase_list)
    if num_phase == 8:
        for p1 in phase_list:
            zeros = [0, 0, 0, 0, 0, 0, 0]
            count = 0
            for p2 in phase_list:
                if p1 == p2:
                    continue
                m1 = p1.split("_")
                m2 = p2.split("_")
                if len(list(set(m1 + m2))) == 3:
                    zeros[count] = 1
                count += 1
            relations.append(zeros)
        relations = np.array(relations).reshape((1, 8, 7))
    else:
        relations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape((1, 4, 3))

    return th.tensor(relations, dtype=th.long)

class MPLight(nn.Module):
    def __init__(self, action_dim):
        super(MPLight, self).__init__()
        self.phase_embedding = nn.Embedding(2, 4)
        self.relation_embedding = nn.Embedding(2, 4)
        self.num_vec_mapping= nn.Linear(1, 4)
        self.lane_embedding= nn.Linear(8, 16)
        self.lane_conv = nn.Linear(32, 20)
        self.relation_conv = nn.Linear(4, 20)
        self.combine_conv = nn.Linear(20, 20)
        self.before_merge = nn.Linear(20, 1)
 
        self.action_dim=action_dim
        self.phase_list=['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL']
        self.list_lane_order= ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"]
        self.register_buffer("phase",th.tensor([[0, 1, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 1],[1, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 1, 0]], dtype=th.float32))
        self.register_buffer("relation",relation(self.phase_list))
    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size=x.shape[0]
        p = self.phase_embedding(x[: ,:8].long())
        p = th.sigmoid(p)

        dic_index = {
            "WL": 0,
            "WT": 1,
            "EL": 3,
            "ET": 4,
            "NL": 6,
            "NT": 7,
            "SL": 9,
            "ST": 10,
        }
        dic_lane = {}


        for i, m in enumerate(
            self.list_lane_order
        ):
            idx = dic_index[m]

            # Original Keras Lambda slice + Dense(4, sigmoid)
            #
            # NOTE:
            # The original code appears to slice one element from
            # feat2 and then pass it through Dense(4).
            # If slice_tensor returns a scalar, this needs reshaping.
            tmp_vec = x[: , idx+8:idx + 9]
            tmp_vec = self.num_vec_mapping(
                tmp_vec
            )

            tmp_vec = th.sigmoid(tmp_vec)

            # Corresponding phase embedding
            tmp_phase = p[:, i, :]
            tmp_phase = tmp_phase

            dic_lane[m] = th.cat(
                [tmp_vec, tmp_phase], dim=1
            )

        list_phase_pressure = []

        for phase in self.phase_list:

            m1, m2 = phase.split("_")

            pressure = (
                th.relu(self.lane_embedding(dic_lane[m1]))
                + th.relu(self.lane_embedding(dic_lane[m2]))
            )

            

            list_phase_pressure.append(pressure)
        relation=self.relation.expand(batch_size,-1, -1)
        relation_embedding=self.relation_embedding(relation)
        

        list_phase_pressure_recomb = []

        for i in range(self.action_dim):
            list_phase_pressure_row = []
            for j in range(self.action_dim):

                if i != j:

                    pair = th.cat(
                        [
                            list_phase_pressure[i],
                            list_phase_pressure[j],
                        ],
                            dim=-1,
                    )
                   
                    # [B, 32]
                    list_phase_pressure_row.append(pair)
            list_phase_pressure_recomb.append(th.stack(list_phase_pressure_row,dim=1))
        
        feature_map=th.stack(list_phase_pressure_recomb,dim=1)
        
        lane_conv=self.lane_conv(feature_map)
        relation_conv=self.relation_conv(relation_embedding)
        combine_feature=lane_conv*relation_conv
        hidden_layer=self.combine_conv(combine_feature)
        before_merge=self.before_merge(hidden_layer)
        
        q_values=before_merge.squeeze(-1).sum(dim=2)
        
        return  q_values





