import torch 
import math
from typing import Optional


class softmax:
    """
    softmax function convert the tensor in probablities 
    """

    def __init__(self):
        pass

    def forward(self,input,axis=-1):
        """
        return the proboblities based on the input tensor

        Arguments:
            input : N*D tensor 
            axis : (int) dimension that we want to do sum 
        return:
            input : N*D (probablities)
        """
        
        exp = torch.exp(input)
        # get the softmax value
        input = exp/torch.sum(exp,axis=-1,keepdim=True)
        return input

class Linear:
    """
    Linear layer 
    """

    def __init__(self,input_dim,output_dim):
        self.weights = torch.rand([input_dim,output_dim],requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights)
        self.bias = torch.zeros(1,output_dim,requires_grad=True)
   
    def forward(self,x):
        x = torch.matmul(x , self.weights)
        self.x =   x + self.bias
        return self.x
    


class Embedding:
    """
     Embedings and positional Embeddings of the input 
    """


    def __init__(self , vocab_size:int , d_model:int , max_seqlen:int) -> None:
        """
        Initialize the postional Embedding vector

        Arguments:
            --vocab_size : (int) vocabulary size
            --d_model : (int) dimension of the model
            --max_seqlen : (int) max token length , reqiured to create the sinsoidal
        Return:
            input embedding and position
        """
        self.vocab_size = vocab_size
        self.d_model = d_model 
        self.max_seqlen = max_seqlen
        self.embedding = torch.nn.Embedding(vocab_size , d_model)
        self.embd_position = torch.nn.Embedding(max_seqlen,d_model)
        # We dont positional embedding to learn
        self.embd_position.requires_grad_(False)
        
    def forward(self, input_tensor : torch.tensor ,position = 0) -> torch.tensor:
        
        """
        forward method to get the embeddings 

        Arguments:
            --input_tensor : (torch.tensor)  input tensor 

        return:
            Embeddings : (torch.tensor) embedding output
        """

        if input_tensor.shape[1] > self.max_seqlen:
            assert "the input token exceeded the max length"
        
        input_tensor = self.embedding(input_tensor)
        # print("it shape",input_tensor.shape)
        positional_embeddings = self.embd_position.weight[position:input_tensor.shape[1]]
        # print("pe",positional_embeddings.shape)
        return input_tensor + positional_embeddings
    

def attention_product(query_states , key_states , value_states , attention_mask = None ,is_decoder:Optional[bool] = False):
    """
    attention dot product 

    arguments:
        --query_states : (torch.Tensor) query tensor
        --key_states : (torch.Tensor) key tensor 
        --value_sates : (torch.Tensor) value states
        --attention_mask : (torch.Tensor) attention mask (for padding , it is must )
        --is_decoder : (bool) based in the boolean , we need to mask the self-attention 

    """  
    q_dim , key_dim = query_states.shape[2] , key_states.shape[2]
    # Need saling factor to reduce the exploding gradient 
    scale_factor = 1 / math.sqrt(query_states.size(-1))
    # atten mask after q . kT , so create tensors on q,k.T shape 
    attn_mask = torch.zeros(q_dim,key_dim,dtype=query_states.dtype)
    if is_decoder:
        #if its decoder we need to mask , forcing decoder attentiont to atten itself 
       # Reference : scaled dot product pytorch
        temp_mask = torch.ones(q_dim , key_dim,dtype=torch.bool).tril(diagonal=0)
        attn_bias = attn_mask.masked_fill_(temp_mask.logical_not(),float("-inf"))
        attn_bias.to(query_states.dtype)
    if attention_mask is not None:
        # mask the PAD value , we dont want our model to atted on PAD topkens
        # R.1 (need to complete)
        pass
    
    # Q @ kT
    kv_weights = query_states @ key_states.transpose(-2,-1) * scale_factor
    # mask the attenntion based in layers
    kv_weights += attn_mask
    kv_weights = softmax().forward(kv_weights)
    # print(kv_weights @ value_states)
    return kv_weights @ value_states