import torch
import logging
from tqdm import tqdm
import torch.nn.utils as utils
from pathlib import Path

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        num_epochs=10,
        warmup_epochs=1,
        grad_accum=2,
        clip_grad_norm=1.0,
        checkpoint_dir='checkpoints'
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.grad_accum = grad_accum
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for i, (inputs, targets) in enumerate(tqdm(self.train_loader, desc="トレーニング")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = self.model.loss_fn(outputs, targets) if hasattr(self.model, 'loss_fn') else torch.nn.functional.cross_entropy(outputs, targets)
            loss = loss / self.grad_accum
            loss.backward()
            
            if (i + 1) % self.grad_accum == 0 or (i + 1) == len(self.train_loader):
                utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * self.grad_accum
        
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="評価"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = self.model.loss_fn(outputs, targets) if hasattr(self.model, 'loss_fn') else torch.nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, accuracy

    def freeze_layers(self, freeze_level=0.3):
        """モデルの一部のレイヤーを凍結"""
        all_params = list(self.model.parameters())
        num_freeze = int(len(all_params) * freeze_level)
        
        for i, param in enumerate(all_params):
            if i < num_freeze:
                param.requires_grad = False

    def train(self):
        best_val_acc = 0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f'エポック {epoch+1}/{self.num_epochs}')
            
            if epoch >= self.num_epochs // 2:
                self.freeze_layers(freeze_level=0.3)
                self.logger.info('モデルの30%のレイヤーを凍結しました')
            
            if epoch < self.warmup_epochs:
                for group in self.optimizer.param_groups:
                    group['lr'] = 1e-4 * (epoch + 1) / self.warmup_epochs
                    
            train_loss = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            self.logger.info(f'トレーニング損失: {train_loss:.4f}')
            self.logger.info(f'検証損失: {val_loss:.4f}, 検証精度: {val_acc:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                         self.checkpoint_dir / 'model_best.pt')
                self.logger.info(f'新しいベストモデルを保存しました (精度: {val_acc:.4f})')
            
            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, self.checkpoint_dir / f'model_epoch_{epoch+1}.pt')
            
            self.scheduler.step() 