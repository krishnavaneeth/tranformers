"""
Attentio layer for encoder and decoder layer 

"""
import torch 
import math 


from typing import Optional
from utils import Linear as linear 
from utils import softmax,attention_product

class attention:

    def __init__(self,dimension,heads,is_decoder:Optional[bool] = False):
        """
        create the attention layers

        arguments:
            --dimension : dimension model (int)
        """
        self.key = linear(dimension,dimension)
        self.query = linear(dimension,dimension)
        self.value = linear(dimension,dimension)
        self.out_proj = linear(dimension,dimension)
        self.dimk = dimension
        self.heads = heads
        self.head_dim = dimension // heads
        self.is_decoder = is_decoder
    
    def _shape(self,input_tensor:torch.Tensor, seq_len , batch_size):
        """
        reshape the Tensor to achieve multi head attention 

        arguments:
            --input_tensor : torch.Tensor (input representation)
            --seq_len : int (token representation of the tensor)
            --batch_size : int (batch size )
        
        return:
            --rehaped tensor 
        """
        # the standard format on multi-head attention 
        # input (1,448,1024) #output (1,16,448,64)  , here the batch size is 1 , num heads 16  embd 1024
        return input_tensor.view(batch_size,seq_len,self.heads,self.head_dim).transpose(1,2)
    
    def forward(self,input , encoder_hidden_states = None):
        """
         to get the attention for the respective input tensor 

         arguments:
            --input : torch.Tensor (input representation)
            --encoer_hidden_states : (torch.tensor ) for decoder cross attention
        return 
            --attention representation for the target input
        """
        if encoder_hidden_states is not None:
            input = encoder_hidden_states
        
        query = self.query.forward(input)
        # print(query)
        key = self.key.forward(input)
        value = self.value.forward(input)
        
        # print(query.shape)
        """
        this is for one attention head"""
        # input = torch.matmul(query,torch.transpose(key,-2,-1)) / self.dimk
        # return torch.matmul(softmax().forward(input) , value)
        """
        Multi-head attention
        """
        # reshape the tensor for standard format for multi-head attention
        # get the sequence length and batch size from the input tensor
        batch_size , seq_len = input.size()[0],input.size()[1]
        query = self._shape(query, seq_len , batch_size)
        value = self._shape(value, seq_len , batch_size)
        key = self._shape(key, seq_len , batch_size)


        # # calculate the attention of Query and Key 
        # attention = torch.matmul(query , key.transpose(2,3))
        
        # attention_qk = softmax().forward(attention)
       
        return attention_product(query_states=query,key_states=key,value_states=value,is_decoder=self.is_decoder).view(batch_size,seq_len,-1)
    
class cross_attention(attention):

    def __init__(self, dimension, heads):
        super().__init__(dimension, heads)
    
    def forward(self,input_tensor,encoder_hidden_states: Optional[torch.Tensor]=None):
        """
        Use for cross attention 

        arguments:
            --input_tensor : input tensor (torch.Tensor)
            --encoder_hidden_states : encoder last hidden state , if its None , call the parent attention 
        
        return:
            cross attention representation
        """
        bsz , seq_len = input_tensor.size(0) , input_tensor.size(1)
        query = self.query.forward(input_tensor)
        query = self._shape(query,seq_len,bsz)
        if encoder_hidden_states is None:
            print("hi :]")
            pass
        
        else:
            print("hello :]")
            scale_factor = 1 / math.sqrt(query.size(-1)) 
            key_value = encoder_hidden_states
            # re-caclulate the seq_len , because we are taking encoder hidden states , so the dim will change
            seq_len_encoder = encoder_hidden_states.size(1)
            # reshape the tensor of muti-head attention
            key = self._shape(self.key.forward(encoder_hidden_states),seq_len_encoder,bsz)
            #query and key.T
            key_value = torch.matmul(query,key.transpose(-2,-1)) * scale_factor
            #apply softmax for the probablity
            key_value = softmax().forward(key_value)

            return torch.matmul(key_value,self._shape(self.value.forward(encoder_hidden_states),seq_len_encoder,bsz)).reshape(bsz,seq_len,-1)
