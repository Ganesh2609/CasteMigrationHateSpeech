import torch 
from torch import nn
from transformers import AutoModelForMaskedLM, logging
logging.set_verbosity_error()


class HateSpeechBERT(nn.Module):

    def __init__(self, num_embeddings:int=1920, num_encodings:int=128, input_encodings:int=2048, num_labels:int=1):

        super(HateSpeechBERT, self).__init__()

        self.transformer = AutoModelForMaskedLM.from_pretrained("ai4bharat/IndicBERTv2-MLM-Back-TLM")
        self.classifier = nn.Linear(in_features=768, out_features=num_embeddings, bias=True)

        self.linear = nn.Sequential(
            self.block(in_features=input_encodings, out_features=4096, dropout=0.3),
            self.block(in_features=4096, out_features=2048, dropout=0.25),
            self.block(in_features=2048, out_features=1024, dropout=0.25),
            self.block(in_features=1024, out_features=512, dropout=0.2),
            self.block(in_features=512, out_features=256, dropout=0.15),
            nn.Linear(in_features=256, out_features=num_encodings)
        )

        self.final_fc = nn.Linear(in_features=2048, out_features=num_labels) 
 
 
    def block(self, in_features: int, out_features: int, dropout: float):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, encodings:torch.Tensor):
        
        transformer_out = self.classifier(self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:, 0, :])
        encodings = self.linear(encodings)
        features = torch.cat([transformer_out, encodings], dim=-1)
        labels = self.final_fc(features)
        return features, labels
    