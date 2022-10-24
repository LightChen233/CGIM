import torch
from torch import nn


class CycleGuidedInteractiveLearningModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block_q = GuidedMultiHeadAttentionModule(args)
        self.block_h = GuidedMultiHeadAttentionModule(args)
        self.block_kb = GuidedMultiHeadAttentionModule(args)

    def forward(self, hidden_state_q, hidden_state_kb, hidden_state_h):
        _, qi_key, qi_value = self.block_q.forward(hidden_state_q)
        hidden_kb, kbi_key, kbi_value = self.block_kb.forward(hidden_state_kb, interact_key=qi_key,
                                                              interact_value=qi_value)
        hidden_h, hi_key, hi_value = self.block_h.forward(hidden_state_h, interact_key=kbi_key,
                                                          interact_value=kbi_value)
        hidden_q, _, _ = self.block_q.forward(hidden_state_q, interact_key=hi_key, interact_value=hi_value)
        return hidden_q, hidden_kb, hidden_h


class GuidedMultiHeadAttentionModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block = TangledBlock(args)
        self.multihead_attn = MultiAttention(args)
        self.linear = QKVLinear(args)

    def forward(self, hidden_state, interact_key=None, interact_value=None):

        query, key, value = self.linear(hidden_state)
        if interact_key is not None and interact_value is not None:
            interact = self.multihead_attn(query, interact_key, interact_value)
            hidden = self.block.forward(query, key, value, interact)
            return hidden, key, value
        else:
            return None, key, value


class TangledBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.multihead_attn = MultiAttention(args)
        self.linear = QKVLinear(args)
        self.hidden_size = args.train.hidden_size
        self.interact_key_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.interact_value_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.inter_mediate = Intermediate(args)
        self.LayerNorm = nn.LayerNorm(args.train.hidden_size, eps=args.train.layer_norm_eps)
        self.output = Output(args)

    def forward(self, query, key, value, interact):
        key = torch.cat([key, interact], dim=2)
        value = torch.cat([value, interact], dim=2)
        k = self.interact_key_linear(key)
        v = self.interact_value_linear(value)
        attn_output = self.multihead_attn(query, k, v)
        norm = self.LayerNorm(attn_output + query + k + v)
        mid_output = self.inter_mediate(norm)

        output = self.output(mid_output, norm)
        return output


class Intermediate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.train.hidden_size, args.train.intermediate_size)
        self.intermediate_act_fn = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.train.intermediate_size, args.train.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.train.hidden_size, eps=args.train.layer_norm_eps)
        self.dropout = nn.Dropout(args.train.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QKVLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.train.hidden_size
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states):
        k = self.linear_k(hidden_states)
        q = self.linear_q(hidden_states)
        v = self.linear_v(hidden_states)
        return q, k, v


class MultiAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.train.hidden_size
        self.num_heads = args.train.num_heads
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=0.4)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output
