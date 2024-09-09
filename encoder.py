"""
Encoder layer for the transformer 
"""

import torch 
from torch import nn
from utils import Linear as linear
from utils import attention_product
from attention import attention


class AsrEncoder:

    def __init__(self,attention_dim:int,fc1_out_dim:int,fc2_out_dim:int,layernorm_dim:int,heads:int):
        """
        encoder layer structure

        arguments:
            --attenyion_dim : attention layers dimension (int)
            --fc1_out_dim : full connected layer dimension (int)
            --fc2_out_dim : full connected layer dimension (int)
            --layernorm_dim : layer normalization dimension (int)
            --head : no of heads in layer (multi-head attention) (int)

        """

        self.attention = attention(dimension=attention_dim,heads=heads)
        self.attention_layer_norm = nn.LayerNorm(attention_dim)
        self.activation = nn.GELU(approximate="none")
        self.fc1 = linear(input_dim=attention_dim,output_dim=fc1_out_dim)
        self.fc2 = linear(input_dim=fc1_out_dim,output_dim=fc2_out_dim)
        self.final_layer_norm = nn.LayerNorm(attention_dim)
    
    def forward(self,input_tensor:torch.Tensor):

        """
        forward propogation for encoder 

        arguments:
            --input_tensor : input tensor (torch.Tensor)
        return:
            encoder hidden states 
        """
        # for add and norm
        reversal = input_tensor

        hidden_states = self.attention.forward(input=input_tensor)
        hidden_states = self.attention_layer_norm(hidden_states)
        hidden_states = hidden_states + reversal
        reversal = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc1.forward(hidden_states)
        hidden_states = self.fc2.forward(hidden_states)
        hidden_states = hidden_states + reversal
        hidden_states = self.final_layer_norm(hidden_states)
        

        return hidden_states
