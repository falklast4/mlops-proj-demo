import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.pytorch
import pandas as pd
from typing import List, Tuple
import argparse
import logging

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data() -> Tuple[List[str], List[int]]:
    """Load training data - replace with your dataset"""
    # Example: Load IMDB sentiment data or any text classification dataset
    # For demo, create sample data
    texts = [
        "I love this movie, it's amazing!",
        "This film is terrible and boring",
        "Great acting and wonderful story",
        "Worst movie I've ever seen",
        # Add more training data here
    ]
    labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    return texts, labels

def train_model(
    texts: List[str], 
    labels: List[int], 
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train DistilBERT model"""
    
    # Initialize MLflow
    mlflow.set_experiment("sentiment-classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_samples", len(texts))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create datasets and dataloaders
        train_dataset = SentimentDataset(X_train, y_train, tokenizer)
        val_dataset = SentimentDataset(X_val, y_val, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # Validation
            model.eval()
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(val_true, val_predictions)
            mlflow.log_metric("val_accuracy", accuracy, step=epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            
            model.train()
        
        # Final evaluation
        model.eval()
        report = classification_report(val_true, val_predictions, output_dict=True)
        
        mlflow.log_metric("final_accuracy", report['accuracy'])
        mlflow.log_metric("final_f1_score", report['macro avg']['f1-score'])
        mlflow.log_metric("final_precision", report['macro avg']['precision'])
        mlflow.log_metric("final_recall", report['macro avg']['recall'])
        
        # Log model
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name="sentiment-classifier"
        )
        
        # Save tokenizer separately
        tokenizer.save_pretrained("./tokenizer")
        mlflow.log_artifacts("./tokenizer", "tokenizer")
        
        print(f"Training completed. Model logged to MLflow.")
        return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    # Load data
    texts, labels = load_data()
    
    # Train model
    model, tokenizer = train_model(
        texts=texts,
        labels=labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )