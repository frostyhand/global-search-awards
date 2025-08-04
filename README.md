# Text Keyword Categorization System

Author:
Tejaswi Suresh: https://www.linkedin.com/in/pseo/

## Overview
This repository contains a sophisticated text processing and categorization system designed for efficiently categorizing large volumes of keywords based on language, content type, and semantic meaning. The solution combines rule-based pattern matching with machine learning classification to achieve high accuracy across diverse categories.

## Key Features
- Fast rule-based categorization using regex patterns
- Machine learning classification for ambiguous cases
- Advanced transformer-based AI categorization for improved accuracy
- Multi-language detection and support for 15+ languages
- Person name identification and validation
- Parallel processing for efficient handling of large datasets
- Comprehensive logging and statistics generation
- Cross-verification between categorization methods
- Ensemble approach combining traditional ML and transformer models

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- tqdm
- torch (for transformer models)
- transformers (Hugging Face transformers library)
- concurrent.futures (standard library)

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib tqdm torch transformers
```

## Directory Structure
The system expects the following directory structure:
```
├── combined_categorization.py     # Main script
├── enhanced_ai_categorization.py  # Transformer-based AI enhancements
├── transformer_integration.py     # Integration of traditional ML and transformer models
├── Reclass/                       # Input directory for keyword files
│   ├── file1.csv
│   └── file2.xlsx
├── Dish/                          # Output directory for categorized files
│   └── Enhanced/                  # Directory for summary files and ML models
│       └── transformer_model/     # Saved transformer model files
├── name_file1.xlsx                # Special person name files (optional)
├── name_file2.xlsx
└── ...
```

## How It Works

### 1. Fast Pattern-Based Categorization
The system first applies rule-based categorization using regex patterns and linguistic rules to quickly classify keywords. This step handles around 60% of all keywords efficiently.

### 2. Machine Learning Classification
For keywords that couldn't be categorized in the first step, the system trains machine learning models (Random Forest and Naive Bayes) using the already categorized keywords as training data.

### 3. Transformer-Based AI Classification
For even higher accuracy, especially with multilingual keywords and ambiguous cases, the system can use a transformer-based model (DistilBERT) that provides:
- Better semantic understanding - Captures word relationships and context
- Improved multilingual capabilities - Better handling of non-English keywords
- Enhanced accuracy - Especially for short or ambiguous keywords
- More robust feature extraction - Less dependent on manually engineered features

### 4. Ensemble Approach
The system can combine predictions from both traditional ML and transformer models through an ensemble approach:
- Using the agreed category when both models predict the same category
- Preferring the transformer prediction when they disagree (due to higher confidence)

### 5. Person Name Identification
A specialized subsystem identifies and categorizes potential person names using multiple approaches:
- Pattern matching against common name structures
- Machine learning classification trained on existing names
- Cross-verification across multiple category files

### 6. Language Detection
The system includes sophisticated language detection for 15+ languages using:
- Character set detection (Unicode ranges)
- Common word patterns
- Script identification

### 7. Cross-Verification
Results from different categorization methods are cross-verified to improve accuracy and resolve conflicts.

## Detailed Function Descriptions

### Main Workflow Functions
- `main()`: Orchestrates the overall categorization process
- `step1_fast_categorization()`: Executes the fast pattern-based categorization
- `step2_ml_categorization()`: Performs machine learning-based categorization
- `process_file_fast()`: Processes individual input files in parallel

### Categorization Functions
- `categorize_keyword_fast()`: Quickly categorizes a keyword using regex patterns
- `categorize_with_patterns()`: More thorough pattern-based categorization
- `categorize_with_ml()`: Applies machine learning models to categorize keywords
- `categorize_with_transformer()`: Uses transformer models for advanced categorization

### Language Detection
- `detect_language()`: Identifies the language of a given keyword
- `is_non_english_language()`: Determines if text contains non-English characters

### Person Name Identification
- `is_likely_person_name()`: Determines if a keyword is likely a person name
- `extract_name_features()`: Extracts features for name classification
- `cross_verify_person_names()`: Cross-verifies potential person names across categories
- `process_person_name_files()`: Processes special files containing person names

### Machine Learning
- `train_ml_model()`: Trains traditional machine learning models for keyword categorization

### Transformer AI Components
- `TransformerKeywordClassifier`: Main class for transformer-based categorization
- `integrate_with_original_script()`: Integrates transformer model with the original script
- `ensemble_categorization()`: Combines traditional ML and transformer predictions
- `evaluate_model_improvements()`: Compares performance of different models

### Special Processing
- `extract_all_chinese_keywords()`: Creates dedicated file for Chinese keywords

## Usage

### Basic Categorization
1. Place input files in the `Reclass` directory (CSV, Excel formats supported)
2. Run the main script:
```bash
python combined_categorization.py
```
3. Categorized keywords will be output to category-specific files in the `Dish` directory
4. Summary files and ML models are saved to `Dish/Enhanced`

### Enhanced AI Categorization
For improved categorization using transformer models:

1. After running the main script, train the transformer model:
```bash
python enhanced_ai_categorization.py
```

2. For ensemble categorization with both traditional ML and transformer models:
```bash
python transformer_integration.py categorize uncategorized_keywords.csv
```

3. To evaluate model performance improvements:
```bash
python transformer_integration.py evaluate
```
This requires an evaluation file (`evaluation_data.csv`) with known true categories.

## Input Format
The script accepts CSV and Excel files with keywords. At minimum, the first column should contain keywords. If a second column exists, it's interpreted as volume/frequency data.

Example input file:
```
Keyword,Volume
example keyword 1,120
example keyword 2,50
...
```

## Output Format
Each category gets its own CSV file with the following structure:
```
Keyword,Volume,Source,Categories
keyword1,120,input_file.csv,"['category1', 'category2']"
keyword2,50,input_file.csv,"['category1']"
...
```

## Customization

### Adding New Categories
To add new categories, update the `CATEGORY_PATTERNS` dictionary with appropriate regex patterns:

```python
CATEGORY_PATTERNS = {
    'existing_category': existing_pattern,
    'new_category': re.compile(r'\bnew_word1\b|\bnew_word2\b', re.IGNORECASE),
}
```

### Language Detection
To improve language detection for specific languages, extend the language-specific patterns:

```python
# Example for improving French detection
FRENCH_ADDITIONAL_TERMS = [
    r'\bterme1\b', r'\bterme2\b', r'\bterme3\b'
]
```

### Customizing Transformer Models
You can customize the transformer model by changing parameters in `enhanced_ai_categorization.py`:

```python
# Use a different pretrained model
transformer_classifier = TransformerKeywordClassifier(model_name="bert-base-multilingual-cased")

# Adjust training parameters
transformer_classifier.train(train_loader, val_loader, epochs=5)
```

### Performance Tuning
For large datasets, adjust the parallelism settings:

```python
# Increase number of parallel workers for faster processing
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
```

## Logging
The script creates detailed log files tracking:
- Processing progress and timing
- Files processed
- Category statistics
- Model training and evaluation metrics
- Errors and warnings

## Machine Learning Models
The system saves trained models in the `Enhanced` directory:
- `keyword_classifier.pkl`: Traditional ML classification model
- `name_classifier.pkl`: Person name classifier model
- `transformer_model/`: Directory containing the transformer model files

These models can be reused for future classification tasks without retraining.

## Future Improvements
Potential enhancements include:

1. **Fine-tuning for specific domains** - Adapting to domain-specific keywords
2. **Multi-label classification** - Handling keywords that belong to multiple categories
3. **Active learning** - Interactive learning from user feedback
4. **Explainable AI** - Providing explanations for why keywords are categorized a certain way
5. **Custom embeddings** - Training custom embeddings for specialized vocabulary
