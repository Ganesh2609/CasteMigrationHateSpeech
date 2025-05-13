import sys
sys.path.append('../Models')

import torch
from torch import nn
from trainer import ModularTrainer
from dataset import get_data_loaders
from xlm_roberta_base import HateSpeechBERT

def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = HateSpeechBERT().to(device)

    root_path = "../No Trans data"

    train_loader = get_data_loaders(text_data=root_path+'/train_processed.csv', feature_data=root_path+'/train_features.csv', batch_size=8)
    test_loader = get_data_loaders(text_data=root_path+'/dev_processed.csv', feature_data=root_path+'/dev_features.csv', batch_size=8)

    learning_rate = 1e-3
    weight_decay = 5e-4
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=False)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        log_path='../Train Data/Logs/xlm_roberta_base.log',
        num_epochs=64,
        checkpoint_path='../Train Data/Checkpoints/xlm roberta base',
        graph_path='../Train Data/Graphs/xlm_roberta_base.png',
        verbose=True,
        device=device 
    )

    trainer.train()
    #trainer.train(resume_from="E:/Work/My Papers/LT-EDI 2025/Caste and Migration Hate/Codebase/Train Data/Checkpoints/indic bert 2 lr3 d4 no trans const reduce/model_epoch_12.pth")

if __name__ == '__main__':
    main()
