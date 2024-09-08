"""
model class 
"""

import torch 
import math 


from Embedding import Embedding
from utils import Linear as linear
from Encoder import AsrEncoder
from Decoder import AsrDecoder

class Model:

    def __init__(self,dmodel:int,head:int,layers:int,vocab_size:int,max_seqlen:int):
        """
        Summary : initilaize the model required parameters 

        Arguments:
            --dmodel : dimension of the model
            --head : attention heads
            --layers : number of encoder and decoder layer
        """
        self.dmodel = dmodel
        self.head = head
        self.layers = layers
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.embedding = Embedding(vocab_size,dmodel,max_seqlen)
        self.encoder_layers = [AsrEncoder(dmodel,1536,dmodel,dmodel,12) for i in range(0,layers)]
        self.decoder_layers = [AsrDecoder(dmodel,1536,dmodel,dmodel,12) for i in range(0,layers)]
        self.final_proj = linear(384,51865)


    def forward(self,input_tensor:torch.Tensor,input_labels:torch.Tensor):
        """
        forward propagation fot enoder layer 

        arguments:
            --input_tensor : input audio
            --input_labels : transcript for the audio
        return:
            list[torch.Tensor] tokens 
        """
        input_labels = self.embedding.forward(input_labels)
        
        for layer in self.encoder_layers:
            input_tensor = layer.forward(input_tensor=input_tensor)

        encoder_hidden_states =  input_tensor
        for layer in self.decoder_layers:
            input_labels = layer.forward(input_tensor=input_labels,encoder_hidden_states=encoder_hidden_states)
        # return the top token , which has high prob
        return torch.topk(self.final_proj.forward(input_labels),k=1).indices
    
if __name__ == "__main__":
    
    model = Model(384,6,1,51865,448)
    audio = torch.rand(1,1500,384) * (1 * math.exp(-5))
    
    transcript = torch.randint(51864,(1,6)) 
    print(model.forward(audio,transcript))