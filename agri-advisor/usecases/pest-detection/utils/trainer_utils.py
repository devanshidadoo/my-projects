import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm.auto import tqdm
import numpy as np
import logging
from config import Config


class EfficientNetB3Trainer:
    def __init__(self):
        # Get all params from Config
        self.num_classes = Config.NUM_CLASSES
        self.device = torch.device(Config.DEVICE)
        self.epochs = Config.EPOCHS
        self.save_best = Config.SAVE_BEST
        self.lr = Config.LR
        self.weight_decay = Config.WEIGHT_DECAY
        self.freeze_features = getattr(Config, "FREEZE_FEATURES", True)

        logging.info("Loading pretrained EfficientNet-B3...")
        # Load model without pretrained weights first
        self.model = models.efficientnet_b3(weights=None)
        
        # Load pretrained weights manually to handle size mismatches
        pretrained_state_dict = models.efficientnet_b3(weights="IMAGENET1K_V1").state_dict()
        
        # Remove the classifier weights since we'll replace it
        pretrained_state_dict.pop('classifier.1.weight', None)
        pretrained_state_dict.pop('classifier.1.bias', None)
        
        # Load pretrained weights (this will ignore missing keys)
        self.model.load_state_dict(pretrained_state_dict, strict=False)
        
        if self.freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace classifier with new one for our number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()) if torch.cuda.is_available() else None
        self.best_val_acc = 0.0

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        class_correct = np.zeros(self.num_classes, dtype=np.int64)
        class_total = np.zeros(self.num_classes, dtype=np.int64)

        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            # Classwise stats
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
            acc = 100.0 * correct / total
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
        train_loss = running_loss / total
        train_acc = correct / total
        classwise_acc = class_correct / np.maximum(class_total, 1)
        return train_loss, train_acc, classwise_acc

    def validate(self, val_loader):
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        class_correct = np.zeros(self.num_classes, dtype=np.int64)
        class_total = np.zeros(self.num_classes, dtype=np.int64)
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validating", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
                # Classwise stats
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1
                temp_acc = 100.0 * val_correct / val_total
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{temp_acc:.2f}%")
        val_loss /= val_total
        val_acc = val_correct / val_total
        classwise_acc = class_correct / np.maximum(class_total, 1)
        return val_loss, val_acc, classwise_acc

    def test(self, test_loader):
        self.model.eval()
        test_correct, test_total = 0, 0
        class_correct = np.zeros(self.num_classes, dtype=np.int64)
        class_total = np.zeros(self.num_classes, dtype=np.int64)
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Testing", leave=False)
            for images, labels in test_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(1)
                test_total += labels.size(0)
                test_correct += preds.eq(labels).sum().item()
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1
                test_acc_temp = 100.0 * test_correct / test_total
                test_bar.set_postfix(acc=f"{test_acc_temp:.2f}%")
        test_acc = test_correct / test_total
        classwise_acc = class_correct / np.maximum(class_total, 1)
        return test_acc, classwise_acc

    def fit(self, train_loader, val_loader, test_loader):
        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch+1}/{self.epochs} - Training started")
            train_loss, train_acc, train_classwise = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_classwise = self.validate(val_loader)
            self.scheduler.step()
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            logging.debug(f"Classwise Val Acc (first 10): {val_classwise[:10]}")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_best)
                logging.info(f"Saved Best Model (Val Acc: {val_acc:.4f})")

        logging.info("Loading Best Model for Testing...")
        self.model.load_state_dict(torch.load(self.save_best, map_location=self.device))
        test_acc, test_classwise = self.test(test_loader)
        logging.info(f"Final Test Accuracy: {test_acc:.4f}")
        logging.debug(f"Classwise Test Acc (first 10): {test_classwise[:10]}")
        return test_acc, test_classwise
