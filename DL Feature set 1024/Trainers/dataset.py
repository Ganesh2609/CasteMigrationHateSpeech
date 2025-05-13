import os
import pandas as pd
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


#tokenizer = AutoTokenizer.from_pretrained("seanbenhur/tanglish-offensive-language-identification")
#tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
#tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-Back-TLM")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
#tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")


class HateSpeechData(Dataset):

    def __init__(self, text_data:str, feature_data:str):

        super(HateSpeechData, self).__init__()

        text_data = pd.read_csv(text_data)
        feature_data = pd.read_csv(feature_data)
        feature_dict = {row["id"]: torch.tensor(row[1:].values, dtype=torch.float32) for _, row in feature_data.iterrows()}
        self.data = []

        for _, row in text_data.iterrows():
            text_id, text, label = row["id"], row["text"], row["label"]
            if text_id not in feature_dict:
                raise ValueError(f"ID {text_id} found in text data but missing in feature data")
            if len(tokenizer(text).input_ids) <= 512:
                self.data.append({"Text": text, "Features": feature_dict[text_id], "Label": label})


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.data[idx]
        

def collate_fn(batch):
    
    text = [item['Text'] for item in batch]
    text = tokenizer(text, return_tensors='pt', padding=True, return_attention_mask=True, return_token_type_ids=False)
    features = torch.stack([item['Features'] for item in batch], dim=0)
    label = torch.tensor([item['Label'] for item in batch], dtype=torch.float32).unsqueeze(dim=1)
    return {
        'Tokens' : text.input_ids,
        'Attention Mask' : text.attention_mask,
        'Features' : features,
        'Label' : label
    }



def get_data_loaders(text_data:str, feature_data:str, batch_size:int=16, num_workers:int=10, prefetch_factor:int=2):
    
    data = HateSpeechData(text_data=text_data, feature_data=feature_data)

    # data_loader = DataLoader(
    #     dataset=data, 
    #     batch_size=batch_size, 
    #     shuffle=True, 
    #     num_workers=num_workers, 
    #     pin_memory=True, 
    #     persistent_workers=True,  
    #     prefetch_factor=prefetch_factor,
    #     collate_fn=collate_fn,
    #     drop_last=True
    # )

    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    return data_loader