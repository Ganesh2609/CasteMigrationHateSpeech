import os
import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import Optional
from logger import TrainingLogger
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


class ModularTrainer:


    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 log_path: Optional[str] = './train data/logs/training.log',
                 num_epochs: Optional[int] = 16,
                 checkpoint_path: Optional[str] = './train data/checkpoints',
                 graph_path: Optional[str] = './train data/graphs/model_loss.png',
                 verbose: Optional[bool] = True,
                 device: Optional[torch.device] = None) -> None:
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.verbose = verbose
        self.loss_update_step = 48

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float(0)

        self.accuracy_score = BinaryAccuracy().to(device)
        self.f1_metric = BinaryF1Score().to(device)

        self.history = {
            'Training Loss': [],
            'Training Accuracy' : [],
            'Training F1 Score' : [],
            'Testing Loss': [],
            'Testing Accuracy' : [],
            'Testing F1 Score' : []
        }

        self.step_history = {
            'Training Loss': [],
            'Training Accuracy' : [],
            'Training F1 Score' : [],
            'Testing Loss': [],
            'Testing Accuracy' : [],
            'Testing F1 Score' : []
        }


    def update_plot(self) -> None:

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].plot(self.step_history['Training Loss'], color='blue', label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(self.step_history['Training Accuracy'], color='purple', label='Training Accuracy')
        axs[0, 1].set_title('Training Accuracy')
        axs[0, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 1].set_ylabel('Error Rate')
        axs[0, 1].legend()

        axs[0, 2].plot(self.step_history['Training F1 Score'], color='green', label='Training F1')
        axs[0, 2].set_title('Training F1 Score')
        axs[0, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 2].set_ylabel('Error Rate')
        axs[0, 2].legend()

        axs[1, 0].plot(self.step_history['Testing Loss'], color='orange', label='Testing Loss')
        axs[1, 0].set_title('Testing Loss')
        axs[1, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        axs[1, 1].plot(self.step_history['Testing Accuracy'], color='brown', label='Testing Accuracy')
        axs[1, 1].set_title('Testing Accuracy')
        axs[1, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 1].set_ylabel('Error Rate')
        axs[1, 1].legend()

        axs[1, 2].plot(self.step_history['Testing F1 Score'], color='red', label='Testing F1')
        axs[1, 2].set_title('Testing F1 Score')
        axs[1, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 2].set_ylabel('Error Rate')
        axs[1, 2].legend()

        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close(fig)

        return


    def train_epoch(self) -> None:

        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        self.f1_metric.reset()

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, batch in t:
                
                input_tokens = batch['Tokens'].to(self.device)
                attention_mask = batch['Attention Mask'].to(self.device)
                features = batch['Features'].to(self.device)
                labels = batch['Label'].to(self.device)

                _, logits = self.model(input_ids=input_tokens, attention_mask=attention_mask, encodings=features)
                loss = self.loss_fn(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                prediction = torch.round(torch.sigmoid(logits))
                acc = self.accuracy_score(prediction, labels).item() * 100

                total_loss += loss.item()
                total_acc += acc
                self.f1_metric.update(prediction, labels)

                self.current_step += 1

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Train Loss' : total_loss/(i+1),
                    'Train Accuracy' : total_acc/(i+1),
                    'Train F1 Score' : self.f1_metric.compute().item()
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i+1))
                    self.step_history['Training Accuracy'].append(total_acc / (i+1))
                    self.step_history['Training F1 Score'].append(self.f1_metric.compute().item())
                    self.update_plot()

        train_loss = total_loss / len(self.train_loader)
        train_acc = total_acc / len(self.train_loader)
        train_f1 = self.f1_metric.compute().item()
        self.history['Training Loss'].append(train_loss)
        self.history['Training Accuracy'].append(train_acc)
        self.history['Training F1 Score'].append(train_f1)
        
        self.logger.info(f"Training loss for epoch {self.current_epoch}: {train_loss}")
        self.logger.info(f"Training accuracy for epoch {self.current_epoch}: {train_acc}")
        self.logger.info(f"Training F1 score for epoch {self.current_epoch}: {train_f1}")

        return
    


    def test_epoch(self) -> None:

        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        self.f1_metric.reset()

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)') as t:
            
            for i, batch in t:
                
                input_tokens = batch['Tokens'].to(self.device)
                attention_mask = batch['Attention Mask'].to(self.device)
                features = batch['Features'].to(self.device)
                labels = batch['Label'].to(self.device)
                
                with torch.no_grad():
                    _, logits = self.model(input_ids=input_tokens, attention_mask=attention_mask, encodings=features)
                    loss = self.loss_fn(logits, labels)
                    prediction = torch.round(torch.sigmoid(logits))
                
                acc = self.accuracy_score(prediction, labels).item() * 100
                total_loss += loss.item()
                total_acc += acc
                self.f1_metric.update(prediction, labels)

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Test Loss' : total_loss/(i+1),
                    'Test Accuracy' : total_acc/(i+1),
                    'Test F1 Score' : self.f1_metric.compute().item()
                })

                if i % self.loss_update_step == 0 and i != 0:

                    self.step_history['Testing Loss'].append(total_loss / (i+1))
                    self.step_history['Testing Accuracy'].append(total_acc / (i+1))
                    self.step_history['Testing F1 Score'].append(self.f1_metric.compute().item())
                    self.update_plot()

        test_loss = total_loss / len(self.test_loader)
        test_acc = total_acc / len(self.test_loader)
        test_f1 = self.f1_metric.compute().item()
        self.history['Testing Loss'].append(test_loss)
        self.history['Testing Accuracy'].append(test_acc)
        self.history['Testing F1 Score'].append(test_f1)

        if self.scheduler:
            self.scheduler.step(test_loss)
            #self.scheduler.step()

        if test_f1 > self.best_metric:
            self.best_metric = test_f1
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Testing loss for epoch {self.current_epoch}: {test_loss}")
        self.logger.info(f"Testing accuracy for epoch {self.current_epoch}: {test_acc}")
        self.logger.info(f"Testing F1 score for epoch {self.current_epoch}: {test_f1}\n")
        if self.scheduler:
            self.logger.info(f"Current Learning rate: {self.scheduler.get_last_lr()}")

        return
    

    def train(self, resume_from: Optional[str]=None) -> None:
        
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.current_epoch}")
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.current_step, 
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch")
    
        print(f"Starting training from epoch {self.current_epoch} to {self.num_epochs}")
        

        for epoch in range(self.current_epoch, self.num_epochs + 1):

            self.current_epoch = epoch
            self.train_epoch()
            
            if self.test_loader:
                self.test_epoch()
    
            self.save_checkpoint()
        
        return
    
    

    def save_checkpoint(self, is_best:Optional[bool]=False):

        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_history' : self.step_history,
            'history': self.history,
            'best_metric': self.best_metric
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(
                self.checkpoint_path, 
                f'model_epoch_{self.current_epoch}.pth'
                #f'model_rewrite.pth'
            )

        torch.save(checkpoint, path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {path}")


    def load_checkpoint(self, checkpoint:Optional[str]=None, resume_from_best:Optional[bool]=False):
        
        if resume_from_best:
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch') + 1
        self.current_step = checkpoint.get('current_step')
        self.best_metric = checkpoint.get('best_metric')
        
        loaded_history = checkpoint.get('history')
        for key in self.history:
            self.history[key] = loaded_history.get(key, self.history[key])

        loaded_step_history = checkpoint.get('step_history')
        for key in self.step_history:
            self.step_history[key] = loaded_step_history.get(key, self.step_history[key])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return
