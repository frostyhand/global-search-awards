import pandas as pd
import os
import logging
import sys
from enhanced_ai_categorization import TransformerKeywordClassifier, categorize_with_transformer
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directories
INPUT_DIR = "Reclass"
OUTPUT_DIR = "Dish"
ENHANCED_DIR = "Dish/Enhanced"
TRANSFORMER_MODEL_DIR = os.path.join(ENHANCED_DIR, "transformer_model")

def ensemble_categorization(uncategorized_file):
    """
    Use both traditional ML and transformer models for ensemble categorization
    """
    # Ensure the uncategorized file exists
    if not os.path.exists(uncategorized_file):
        logger.error(f"Uncategorized file {uncategorized_file} not found")
        return
    
    # Load uncategorized data
    uncategorized_df = pd.read_csv(uncategorized_file)
    logger.info(f"Loaded {len(uncategorized_df)} uncategorized keywords")
    
    # Check if the traditional ML model exists
    ml_model_path = os.path.join(ENHANCED_DIR, "keyword_classifier.pkl")
    if not os.path.exists(ml_model_path):
        logger.warning("Traditional ML model not found. Run the original script first.")
    else:
        # Load the traditional ML model
        try:
            ml_classifier = joblib.load(ml_model_path)
            logger.info("Traditional ML model loaded successfully")
            
            # Get predictions from ML model (assuming it's a sklearn pipeline)
            ml_predictions = ml_classifier.predict(uncategorized_df['Keyword'])
            uncategorized_df['ML_Category'] = ml_predictions
            logger.info("Traditional ML categorization complete")
        except Exception as e:
            logger.error(f"Error using traditional ML model: {e}")
    
    # Check if transformer model exists
    if not os.path.exists(TRANSFORMER_MODEL_DIR):
        logger.warning("Transformer model not found. Run enhanced_ai_categorization.py first.")
    else:
        # Use transformer model for categorization
        try:
            uncategorized_df = categorize_with_transformer(uncategorized_df, TRANSFORMER_MODEL_DIR)
            logger.info("Transformer categorization complete")
        except Exception as e:
            logger.error(f"Error using transformer model: {e}")
    
    # Create ensemble prediction by combining both models
    if 'ML_Category' in uncategorized_df.columns and 'TransformerCategory' in uncategorized_df.columns:
        # Simple ensemble: if the models agree, use their prediction
        # If they disagree, prioritize the transformer prediction (higher confidence)
        uncategorized_df['EnsembleCategory'] = uncategorized_df.apply(
            lambda row: row['ML_Category'] if row['ML_Category'] == row['TransformerCategory'] 
            else row['TransformerCategory'], axis=1
        )
        
        # Count agreements
        agreement_count = (uncategorized_df['ML_Category'] == uncategorized_df['TransformerCategory']).sum()
        agreement_pct = agreement_count / len(uncategorized_df) * 100
        logger.info(f"Models agree on {agreement_count} keywords ({agreement_pct:.2f}%)")
        
        # Save results with all predictions
        results_file = "ensemble_categorized.csv"
        uncategorized_df.to_csv(results_file, index=False)
        logger.info(f"Saved ensemble results to {results_file}")
        
        # Create categorized files for each category in the ensemble prediction
        categories = uncategorized_df['EnsembleCategory'].unique()
        for category in categories:
            # Filter for this category
            category_df = uncategorized_df[uncategorized_df['EnsembleCategory'] == category]
            # Save to category-specific file
            category_file = os.path.join(OUTPUT_DIR, f"{category}.csv")
            
            # If file exists, append, otherwise create
            if os.path.exists(category_file):
                # Load existing file
                existing_df = pd.read_csv(category_file)
                # Append new keywords
                combined_df = pd.concat([existing_df, category_df[['Keyword']]], ignore_index=True)
                # Remove duplicates
                combined_df.drop_duplicates(subset=['Keyword'], inplace=True)
                # Save
                combined_df.to_csv(category_file, index=False)
            else:
                # Just save the keywords column
                category_df[['Keyword']].to_csv(category_file, index=False)
                
            logger.info(f"Added {len(category_df)} keywords to category '{category}'")
    
    else:
        logger.warning("Either traditional ML or transformer predictions missing. Cannot create ensemble.")
        # Save whatever predictions we have
        results_file = "partial_categorized.csv"
        uncategorized_df.to_csv(results_file, index=False)
        logger.info(f"Saved partial results to {results_file}")

def evaluate_model_improvements():
    """
    Compare performance of traditional ML vs transformer model
    """
    # Check if we have ground truth data for evaluation
    evaluation_file = "evaluation_data.csv"
    if not os.path.exists(evaluation_file):
        logger.warning("No evaluation data found. Create a file with 'Keyword' and 'TrueCategory' columns.")
        return
    
    # Load evaluation data
    eval_df = pd.read_csv(evaluation_file)
    if 'Keyword' not in eval_df.columns or 'TrueCategory' not in eval_df.columns:
        logger.warning("Evaluation data must have 'Keyword' and 'TrueCategory' columns.")
        return
    
    logger.info(f"Evaluating with {len(eval_df)} labeled examples")
    
    # Get predictions from traditional ML model
    ml_model_path = os.path.join(ENHANCED_DIR, "keyword_classifier.pkl")
    if os.path.exists(ml_model_path):
        try:
            ml_classifier = joblib.load(ml_model_path)
            ml_predictions = ml_classifier.predict(eval_df['Keyword'])
            eval_df['ML_Category'] = ml_predictions
            
            # Calculate accuracy
            ml_accuracy = (eval_df['ML_Category'] == eval_df['TrueCategory']).mean() * 100
            logger.info(f"Traditional ML model accuracy: {ml_accuracy:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating traditional ML model: {e}")
    
    # Get predictions from transformer model
    if os.path.exists(TRANSFORMER_MODEL_DIR):
        try:
            eval_df = categorize_with_transformer(eval_df, TRANSFORMER_MODEL_DIR)
            
            # Calculate accuracy
            transformer_accuracy = (eval_df['TransformerCategory'] == eval_df['TrueCategory']).mean() * 100
            logger.info(f"Transformer model accuracy: {transformer_accuracy:.2f}%")
            
            # Calculate improvement
            if 'ML_Category' in eval_df.columns:
                improvement = transformer_accuracy - ml_accuracy
                logger.info(f"Transformer improves accuracy by {improvement:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating transformer model: {e}")
    
    # Evaluate ensemble if both predictions exist
    if 'ML_Category' in eval_df.columns and 'TransformerCategory' in eval_df.columns:
        # Create ensemble prediction
        eval_df['EnsembleCategory'] = eval_df.apply(
            lambda row: row['ML_Category'] if row['ML_Category'] == row['TransformerCategory'] 
            else row['TransformerCategory'], axis=1
        )
        
        # Calculate accuracy
        ensemble_accuracy = (eval_df['EnsembleCategory'] == eval_df['TrueCategory']).mean() * 100
        logger.info(f"Ensemble model accuracy: {ensemble_accuracy:.2f}%")
        
        # Save evaluation results
        eval_df.to_csv("model_evaluation_results.csv", index=False)
        logger.info("Saved detailed evaluation results")

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ENHANCED_DIR, exist_ok=True)
    
    # Process command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "categorize":
            # Categorize uncategorized keywords
            if len(sys.argv) > 2:
                uncategorized_file = sys.argv[2]
            else:
                uncategorized_file = "uncategorized_keywords.csv"
            ensemble_categorization(uncategorized_file)
        
        elif sys.argv[1] == "evaluate":
            # Evaluate model performance
            evaluate_model_improvements()
        
        else:
            print("Usage:")
            print("  python transformer_integration.py categorize [file.csv]")
            print("  python transformer_integration.py evaluate")
    else:
        print("Usage:")
        print("  python transformer_integration.py categorize [file.csv]")
        print("  python transformer_integration.py evaluate") 