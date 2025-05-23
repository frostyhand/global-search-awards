import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KeywordDataset(Dataset):
    """Custom PyTorch dataset for keyword categorization"""
    def __init__(self, texts, labels=None, tokenizer=None, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Convert to the format expected by PyTorch (squeeze batch dimension)
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

class TransformerKeywordClassifier:
    """Transformer-based classifier for keyword categorization"""
    def __init__(self, model_name="distilbert-base-uncased", num_labels=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, df, text_col='Keyword', label_col='ML_Category'):
        """Prepare data for training"""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(df[label_col])
        
        # Update num_labels
        self.num_labels = len(self.label_encoder.classes_)
        logger.info(f"Number of classes: {self.num_labels}")
        
        # Split into train/val
        texts = df[text_col].values
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.1, random_state=42, stratify=labels
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = KeywordDataset(X_train, y_train, self.tokenizer)
        val_dataset = KeywordDataset(X_val, y_val, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        return train_loader, val_loader, X_val, y_val
    
    def train(self, train_loader, val_loader, epochs=3):
        """Train the transformer model"""
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_accuracy = 0
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_accuracy += (preds == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = val_accuracy / len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Save best model
                torch.save(self.model.state_dict(), "best_transformer_model.pth")
        
        # Load best model
        self.model.load_state_dict(torch.load("best_transformer_model.pth"))
        logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
        
        return best_accuracy
    
    def evaluate(self, X_val, y_val):
        """Evaluate model on validation data"""
        self.model.eval()
        
        # Create dataset and dataloader
        val_dataset = KeywordDataset(X_val, None, self.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Get predictions
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(
            y_val, 
            all_preds,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        
        logger.info("Classification Report:\n" + report)
        return report
    
    def predict(self, texts):
        """Predict categories for new texts"""
        self.model.eval()
        
        # Create dataset and dataloader
        dataset = KeywordDataset(texts, None, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16)
        
        # Get predictions
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        # Convert numerical predictions to category names
        predicted_categories = self.label_encoder.inverse_transform(all_preds)
        return predicted_categories
    
    def save(self, path="transformer_classifier"):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(path, "label_encoder.pkl"))
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path="transformer_classifier"):
        """Load saved model and tokenizer"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        
        # Load label encoder
        self.label_encoder = joblib.load(os.path.join(path, "label_encoder.pkl"))
        self.num_labels = len(self.label_encoder.classes_)
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Number of classes: {self.num_labels}")

def integrate_with_original_script(output_dir, enhanced_dir):
    """Function to integrate transformer model with the original script"""
    # Read all categorized data
    all_categorized_data = []
    category_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    
    if not category_files:
        logger.warning("No category files found for transformer training")
        return
    
    # Load data from each category file
    for category_file in category_files:
        category_name = category_file.replace('.csv', '')
        
        try:
            category_df = pd.read_csv(os.path.join(output_dir, category_file))
            if 'Keyword' not in category_df.columns:
                logger.warning(f"Invalid format in {category_file}, skipping")
                continue
                
            # Add category as a column
            category_df['ML_Category'] = category_name
            
            # Add to combined training data
            all_categorized_data.append(category_df[['Keyword', 'ML_Category']])
            
        except Exception as e:
            logger.error(f"Error reading {category_file}: {e}")
    
    if not all_categorized_data:
        logger.warning("No valid category data found for transformer training")
        return
        
    # Combine all categorized data
    training_df = pd.concat(all_categorized_data, ignore_index=True)
    
    # Remove duplicates
    training_df.drop_duplicates(subset=['Keyword'], keep='first', inplace=True)
    
    logger.info(f"Training transformer model with {len(training_df)} keywords")
    
    # Check if we have enough data
    if len(training_df) < 200:
        logger.warning(f"Only {len(training_df)} examples. Need more data for transformer model.")
        return
    
    # Initialize transformer classifier
    transformer_classifier = TransformerKeywordClassifier()
    
    # Prepare data
    train_loader, val_loader, X_val, y_val = transformer_classifier.prepare_data(training_df)
    
    # Train model
    transformer_classifier.train(train_loader, val_loader, epochs=3)
    
    # Evaluate model
    transformer_classifier.evaluate(X_val, y_val)
    
    # Save model
    save_path = os.path.join(enhanced_dir, "transformer_model")
    transformer_classifier.save(save_path)
    
    logger.info("Transformer model training complete")

def categorize_with_transformer(uncategorized_df, model_path):
    """Use transformer model to categorize keywords"""
    # Load model
    transformer_classifier = TransformerKeywordClassifier()
    transformer_classifier.load(model_path)
    
    # Get predictions
    keywords = uncategorized_df['Keyword'].values
    predicted_categories = transformer_classifier.predict(keywords)
    
    # Add predictions to dataframe
    uncategorized_df['TransformerCategory'] = predicted_categories
    
    logger.info(f"Categorized {len(uncategorized_df)} keywords with transformer model")
    
    return uncategorized_df

# Example usage
if __name__ == "__main__":
    # Define directories as in the original script
    OUTPUT_DIR = "Dish"
    ENHANCED_DIR = "Dish/Enhanced"
    
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ENHANCED_DIR, exist_ok=True)
    
    # Integrate with original script
    integrate_with_original_script(OUTPUT_DIR, ENHANCED_DIR)
    
    # Example of using the transformer for inference
    # Load uncategorized data
    uncategorized_file = "uncategorized_keywords.csv"
    if os.path.exists(uncategorized_file):
        uncategorized_df = pd.read_csv(uncategorized_file)
        model_path = os.path.join(ENHANCED_DIR, "transformer_model")
        
        if os.path.exists(model_path):
            # Categorize with transformer
            results_df = categorize_with_transformer(uncategorized_df, model_path)
            
            # Save results
            results_df.to_csv("transformer_categorized.csv", index=False)
            logger.info("Saved transformer categorization results") 