import pandas as pd
import os
import re
import time
import logging
import concurrent.futures
from pathlib import Path
import glob
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# Common first names for person name detection
COMMON_FIRST_NAMES = {
    'firstname1', 'firstname2', 'firstname3', 'firstname4', 'firstname5',
    'firstname6', 'firstname7', 'firstname8', 'firstname9', 'firstname10',
    # More placeholder first names would go here
    'firstname11', 'firstname12', 'firstname13', 'firstname14', 'firstname15'
}

# Common last names for person name detection
COMMON_LAST_NAMES = {
    'lastname1', 'lastname2', 'lastname3', 'lastname4', 'lastname5',
    'lastname6', 'lastname7', 'lastname8', 'lastname9', 'lastname10',
    # More placeholder last names would go here
    'lastname11', 'lastname12', 'lastname13', 'lastname14', 'lastname15'
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_categorization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directories
INPUT_DIR = "Reclass"
OUTPUT_DIR = "Dish"
ENHANCED_DIR = "Dish/Enhanced"
PERSON_SOURCES_DIR = "." # Directory where the name-specific Excel files are located

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

# Special Excel files containing person names
PERSON_NAME_FILES = [
    "name_file1.xlsx",
    "name_file2.xlsx", 
    "name_file3.xlsx",
    "name_file4.xlsx"
]

# Define category mappings for person name files
PERSON_FILE_CATEGORIES = {
    "name_file1.xlsx": ["person_names", "category1"],
    "name_file2.xlsx": ["person_names"],
    "name_file3.xlsx": ["person_names", "category2"],
    "name_file4.xlsx": ["person_names", "category3", "category4"]
}

# Expanded ethnic terms
EXPANDED_ETHNICITIES = [
    r'\bethnic1\b', r'\bethnic2\b', r'\bethnic3\b', r'\bethnic4\b', r'\bethnic5\b', 
    r'\bethnic6\b', r'\bethnic7\b', r'\bethnic8\b', r'\bethnic9\b', r'\bethnic10\b', 
    r'\bethnic11\b', r'\bethnic12\b', r'\bethnic13\b', r'\bethnic14\b', r'\bethnic15\b', 
    r'\bethnic16\b', r'\bethnic17\b', r'\bethnic18\b', r'\bethnic19\b', r'\bethnic20\b',
    r'\bnationality1\b', r'\bnationality2\b', r'\bnationality3\b', r'\bnationality4\b', r'\bnationality5\b',
    r'\bnationality6\b', r'\bnationality7\b', r'\bnationality8\b', r'\bnationality9\b', r'\bnationality10\b',
    r'\bregion1\b', r'\bregion2\b', r'\bregion3\b', r'\bregion4\b', r'\bregion5\b',
    r'\bregion6\b', r'\bregion7\b', r'\bregion8\b', r'\bregion9\b', r'\bregion10\b'
]

# Expanded website terms
EXPANDED_WEBSITE_TERMS = [
    r'\bwww\b', r'\bcom\b', r'\bnet\b', r'\borg\b', r'\bedu\b', r'\bgov\b', r'\bmil\b',
    r'\bco\b', r'\bme\b', r'\binfo\b', r'\bbiz\b', r'\bname\b', r'\bpro\b', r'\bint\b',
    r'\bhttp\b', r'\bhttps\b', r'\bftp\b', r'\bsite\b', r'\bwebsite\b', r'\bweb\b',
    r'\bpage\b', r'\bpages\b', r'\burl\b', r'\blink\b', r'\blinks\b', r'\bdomain\b',
    r'\bdomains\b', r'\bhost\b', r'\bhosting\b', r'\bserver\b', r'\bservers\b',
    r'\bip\b', r'\bipv4\b', r'\bipv6\b', r'\bdns\b', r'\bssl\b', r'\btls\b',
    r'\bhtml\b', r'\bhtm\b', r'\bphp\b', r'\basp\b', r'\bjsp\b', r'\bcgi\b',
    r'\bxml\b', r'\bjson\b', r'\bcss\b', r'\bjs\b', r'\bjavascript\b',
    r'\bsite1\b', r'\bsite2\b', r'\bsite3\b', r'\bsite4\b', r'\bsite5\b',
    r'\bweb1\b', r'\bweb2\b', r'\bweb3\b', r'\bweb4\b', r'\bweb5\b'
]

# Expanded category terms
EXPANDED_CATEGORY = [
    r'\bcategory1\b', r'\bcategory2\b', r'\bcategory3\b', r'\bcategory4\b', r'\bcategory5\b', 
    r'\bcategory6\b', r'\bcategory7\b', r'\bcategory8\b', r'\bcategory9\b', r'\bcategory10\b', 
    r'\bcategory11\b', r'\bcategory12\b', r'\bcategory13\b', r'\bcategory14\b', r'\bcategory15\b',
    r'\bterm1\b', r'\bterm2\b', r'\bterm3\b', r'\bterm4\b', r'\bterm5\b',
    r'\bterm6\b', r'\bterm7\b', r'\bterm8\b', r'\bterm9\b', r'\bterm10\b'
]

# Language patterns
JAPANESE_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
KOREAN_PATTERN = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]')
CHINESE_PATTERN = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\u2CEB0-\u2EBEF\u30000-\u3134F\uF900-\uFAFF]')
THAI_PATTERN = re.compile(r'[\u0E00-\u0E7F]')
INDIAN_PATTERN = re.compile(r'[\u0900-\u097F\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]')
TURKISH_PATTERN = re.compile(r'[\u00C7\u00E7\u011E\u011F\u0130\u0131\u00D6\u00F6\u015E\u015F\u00DC\u00FC]')
MIDDLE_EASTERN_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u1EE00-\u1EEFF]')
RUSSIAN_PATTERN = re.compile(r'[\u0400-\u04FF\u0500-\u052F]')
SPANISH_PATTERN = re.compile(r'[\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1\u00C1\u00C9\u00CD\u00D3\u00DA\u00D1]|(?:\b(?:el|la|los|las|un|una|unos|unas|lo|al|del)\b)')
PORTUGUESE_PATTERN = re.compile(r'[\u00E3\u00F5\u00E2\u00EA\u00F4\u00E0\u00E1\u00E9\u00ED\u00F3\u00FA\u00E7\u00C3\u00D5\u00C2\u00CA\u00D4\u00C0\u00C1\u00C9\u00CD\u00D3\u00DA\u00C7]|(?:\b(?:o|a|os|as|um|uma|uns|umas|ao|do|da|no|na|nos|nas)\b)')
EAST_EUROPEAN_PATTERN = re.compile(r'[\u0100-\u017F]')
ASIAN_PATTERN = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u3131-\u318E\uAC00-\uD7AF\u0E00-\u0E7F]')

# Define language patterns for detection
URL_PATTERN = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', re.IGNORECASE)
CONTENT_INTEREST_PATTERN = re.compile(r'\bcontent_interest\b|\bcontent_relevant\b|\bcontent_of_interest\b|\bspecial_content\b|\bflagged_content\b', re.IGNORECASE)
EMOJI_PATTERN = re.compile(r'[\U0001F000-\U0001F9FF\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]')

# Define category patterns for fast categorization
CATEGORY_PATTERNS = {
    'spanish': re.compile(r'\bespanol\b|\bespanola\b|\bespanolas\b|\bespanoles\b|\blatino\b|\blatina\b|\blatinas\b|\blatinos\b', re.IGNORECASE),
    'portuguese': re.compile(r'\bportuguesa\b|\bportuguesas\b|\bportugues\b|\bportugueses\b|\bbrasil\b|\bbrazil\b|\bbrasilian\b|\bbrazilian\b', re.IGNORECASE),
    'french': re.compile(r'\bfrancais\b|\bfrancaise\b|\bfrancaises\b|\bfrench\b|\bfrance\b', re.IGNORECASE),
    'german': re.compile(r'\bdeutsch\b|\bdeutsche\b|\bdeutsches\b|\bdeutschen\b|\bgerman\b|\bgermany\b', re.IGNORECASE),
    'italian': re.compile(r'\bitaliano\b|\bitaliana\b|\bitaliane\b|\bitaliani\b|\bitaly\b|\bitalian\b', re.IGNORECASE),
    'russian': re.compile(r'\brussian\b|\brussia\b|\brussians\b|\brusskiy\b|\brusskaya\b|\brusskoe\b|\brusskoye\b', re.IGNORECASE),
    'japanese': re.compile(r'\bjapan\b|\bjapanese\b|\bjapan\b|\bcontent_term1\b', re.IGNORECASE),
    'korean': re.compile(r'\bkorean\b|\bkorea\b', re.IGNORECASE),
    'chinese': re.compile(r'\bchinese\b|\bchina\b', re.IGNORECASE),
    'thai': re.compile(r'\bthai\b|\bthailand\b', re.IGNORECASE),
    'turkish': re.compile(r'\bturkish\b|\bturkey\b', re.IGNORECASE),
    'indonesian': re.compile(r'\bindonesian\b|\bindonesia\b|\bcontent_term2\b', re.IGNORECASE),
    'vietnamese': re.compile(r'\bvietnam\b|\bvietnamese\b', re.IGNORECASE),
    'middle_eastern': re.compile(r'\barab\b|\barabic\b|\barabs\b|\barabian\b|\bmiddle east\b|\bmiddle eastern\b', re.IGNORECASE),
    'east_european': re.compile(r'\beast\b|\beastern\b|\beurope\b|\beuropean\b', re.IGNORECASE),
    'western_european': re.compile(r'\bwestern\b|\bwest\b|\beurope\b|\beuropean\b', re.IGNORECASE),
}

# Function to detect language with modified Vietnamese detection
def detect_language(keyword):
    """Detect language of a keyword using regex patterns"""
    if not isinstance(keyword, str) or not keyword.strip():
        return None
    
    # Check for Amharic characters
    if re.search(r'[\u1200-\u137F]', keyword):
        return 'misc'
        
    # Check Vietnamese first with expanded terms
    if any(re.search(pattern, keyword, re.IGNORECASE) for pattern in EXPANDED_VIETNAMESE):
        return 'vietnamese'
        
    # Japanese has unique characters
    if JAPANESE_PATTERN.search(keyword):
        return 'japanese'
        
    # Korean has unique characters
    if KOREAN_PATTERN.search(keyword):
        return 'korean'
        
    # Chinese has unique characters
    if CHINESE_PATTERN.search(keyword):
        return 'chinese'
        
    # Thai has unique characters
    if THAI_PATTERN.search(keyword):
        return 'thai'
        
    # Burmese (added to Asian terms)
    if re.search(r'[\u1000-\u109F]', keyword):
        return 'asian_terms'
        
    # Indian languages have unique scripts
    if INDIAN_PATTERN.search(keyword):
        return 'indian_terms'
        
    # Turkish has unique characters (excluding "girls" from being classified as Turkish)
    if TURKISH_PATTERN.search(keyword) and not re.search(r'\bgirls\b', keyword, re.IGNORECASE):
        return 'turkish'
        
    # Middle Eastern languages have unique scripts
    if MIDDLE_EASTERN_PATTERN.search(keyword):
        return 'middle_eastern'
        
    # Russian has unique script
    if RUSSIAN_PATTERN.search(keyword):
        return 'russian'
        
    # Spanish has some unique characters and common words
    if SPANISH_PATTERN.search(keyword):
        return 'spanish'
        
    # Portuguese has some unique characters and common words
    if PORTUGUESE_PATTERN.search(keyword):
        return 'portuguese'
        
    # Greek pattern - return 'greek' instead of 'western_european'
    if re.search(r'[αβγδεζηθικλμνξοπρσςτυφχψω]', keyword, re.IGNORECASE):
        return 'greek'
        
    # Nordic languages (Danish, Swedish, Finnish, Norwegian)
    if re.search(r'\bdansk\b|\bdanish\b|\bsvensk\b|\bswedish\b|\bfinsk\b|\bfinnish\b|\bnorsk\b|\bnorwegian\b|\bköpenhamn\b|\bstockholm\b|\boslo\b|\bhelsinki\b', keyword, re.IGNORECASE):
        return 'nordic'
        
    # Italian (check common Italian words)
    if re.search(r'\b(il|lo|la|i|gli|le|di|da|in|con|su|per|tra|fra)\b', keyword, re.IGNORECASE):
        return 'western_european'
        
    # Asian catch-all if not detected as specific Asian language
    if ASIAN_PATTERN.search(keyword):
        return 'asian_terms'
        
    # Eastern European last to prevent false positives
    if EAST_EUROPEAN_PATTERN.search(keyword):
        return 'east_european'
    
    return None

def categorize_keyword_fast(keyword):
    """Efficiently categorize a keyword using pre-compiled regex patterns"""
    if not isinstance(keyword, str) or not keyword.strip():
        return []
    
    categories = set()
    
    # Check for website pattern - only for common website patterns
    if re.search(r'\b(website1|website2|website3|website4)\b', keyword, re.IGNORECASE):
        categories.add('website_terms')
    
    # Check for Spanish characters
    if re.search(r'[áó]', keyword):
        categories.add('spanish')
    
    # Check URL pattern
    if URL_PATTERN.search(keyword):
        categories.add('website_terms')
    
    # Check all category patterns
    for category, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(keyword):
            categories.add(category)
    
    # Check for person names
    if is_likely_person_name(keyword):
        categories.add('person_names')
    
    # Check language patterns with the improved detect_language function
    language = detect_language(keyword)
    if language:
        categories.add(language)
    
    # Handle specific rules
    
    # Rule 30: ALL keywords with "programa" should be in service_category and spanish
    if re.search(r'\bprograma\b', keyword, re.IGNORECASE):
        categories.add('spanish')
        categories.add('service_category')
    
    # Remove Vietnamese from east_european
    if 'vietnamese' in categories and 'east_european' in categories:
        categories.remove('east_european')
    
    # Remove Spanish from east_european
    if 'spanish' in categories and 'east_european' in categories:
        categories.remove('east_european')
    
    # Remove Portuguese from east_european
    if 'portuguese' in categories and 'east_european' in categories:
        categories.remove('east_european')
    
    # Always add category1 for certain keywords
    if re.search(r'\bterm1\b|\bterm2\b|\bterm3\b|\bterm4\b', keyword, re.IGNORECASE):
        categories.add('category1')
    
    # College, school, party = role_play
    if re.search(r'college|school|party', keyword, re.IGNORECASE):
        categories.add('role_play')
    
    # Specific word patterns for various categories
    if re.search(r'term1|term2|term3', keyword, re.IGNORECASE):
        categories.add('category2')
    
    if re.search(r'term4|term5|term6', keyword, re.IGNORECASE):
        categories.add('category3')
    
    if re.search(r'term7|term8|term9', keyword, re.IGNORECASE):
        categories.add('orientation')
    
    if re.search(r'term10|term11|term12', keyword, re.IGNORECASE):
        categories.add('category4')
    
    # Men, man, male, other orientation terms
    if re.search(r'men|man|male|males|mens|woman|women|term13|term14|term15', keyword, re.IGNORECASE):
        categories.add('orientation')
    
    # Various specialized category terms
    if re.search(r'term16|term17|term18', keyword, re.IGNORECASE):
        categories.add('category5')
    
    if re.search(r'bear|teddy', keyword, re.IGNORECASE):
        categories.add('body_types')

    # --- Custom rules ---
    # 1. Any keyword containing how, what, when, who, why = review
    if re.search(r'\b(how|what|when|who|why)\b', keyword, re.IGNORECASE):
        categories.add('review')

    # 2. Category-specific terms
    if re.search(r'\bterm19\b|\bterm20\b', keyword, re.IGNORECASE):
        categories.add('category6')

    # 3. Role-specific terms
    if re.search(r'\bterm21\b|\bterm22\b', keyword, re.IGNORECASE):
        categories.add('role_play')
    
    return list(categories)

def process_file_fast(file_path):
    """Process a single CSV file using the fast categorization method"""
    try:
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        
        # Standardize column names
        if len(df.columns) > 0:
            df.rename(columns={df.columns[0]: 'Keyword'}, inplace=True)
            
            # Ensure Volume column exists
            if 'Volume' not in df.columns and len(df.columns) > 1:
                df.rename(columns={df.columns[1]: 'Volume'}, inplace=True)
            if 'Volume' not in df.columns:
                df['Volume'] = 1
                
            # Clean up keyword column
            df['Keyword'] = df['Keyword'].astype(str).str.strip()
            
            # Add source column
            df['Source'] = file_name
            
            # Apply fast categorization to each keyword
            df['Categories'] = df['Keyword'].apply(categorize_keyword_fast)
            
            # Filter for uncategorized keywords (empty category list)
            uncategorized_df = df[df['Categories'].apply(lambda x: len(x) == 0)]
            
            # Filter for categorized keywords (non-empty category list)
            categorized_df = df[df['Categories'].apply(lambda x: len(x) > 0)]
            
            # Save categorized data to respective category files
            for _, row in categorized_df.iterrows():
                for category in row['Categories']:
                    category_file = os.path.join(OUTPUT_DIR, f"{category}.csv")
                    
                    # Create category file if it doesn't exist
                    if not os.path.exists(category_file):
                        pd.DataFrame(columns=['Keyword', 'Volume', 'Source', 'Categories']).to_csv(category_file, index=False)
                    
                    # Append row to category file
                    row_df = pd.DataFrame({
                        'Keyword': [row['Keyword']],
                        'Volume': [row['Volume']],
                        'Source': [row['Source']],
                        'Categories': [row['Categories']]
                    })
                    row_df.to_csv(category_file, mode='a', header=False, index=False)
            
            return uncategorized_df
        else:
            logger.warning(f"Empty or invalid file: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

def step1_fast_categorization():
    """
    Step 1: Quickly categorize keywords based on patterns and rules,
    saving categorized keywords to respective category files
    """
    logger.info("Starting Step 1: Fast categorization")
    start_time = time.time()
    
    # Find and process all input files in parallel
    input_files = []
    uncategorized_dfs = []
    
    # Find all Excel files in the input directory
    for ext in ['*.xlsx', '*.xls', '*.csv']:
        input_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not input_files:
        logger.warning(f"No input files found in {INPUT_DIR}")
        return None
    
    logger.info(f"Found {len(input_files)} input files to process")
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(input_files))) as executor:
        # Submit all file processing tasks
        future_to_file = {executor.submit(process_file_fast, file_path): file_path for file_path in input_files}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(input_files), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    uncategorized_dfs.append(result)
                    logger.info(f"Processed {file_path} with {len(result)} uncategorized keywords")
                else:
                    logger.info(f"Processed {file_path} (all keywords categorized or empty file)")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    # Calculate processing time
    processing_time = time.time() - start_time
    logger.info(f"Fast categorization completed in {processing_time/60:.2f} minutes")
    
    # Generate detailed category statistics
    category_stats = {}
    for category_file in glob.glob(os.path.join(OUTPUT_DIR, "*.csv")):
        category_name = os.path.basename(category_file).replace('.csv', '')
        try:
            df = pd.read_csv(category_file)
            count = len(df)
            category_stats[category_name] = count
        except Exception as e:
            logger.error(f"Error reading {category_file}: {e}")
    
    # Log category statistics
    if category_stats:
        logger.info("Category statistics from fast categorization:")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count} keywords")
    
    # Write fast categorization summary to file
    with open(os.path.join(ENHANCED_DIR, "fast_categorization_summary.txt"), "w", encoding='utf-8') as f:
        f.write(f"Fast Categorization Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Files processed: {len(input_files)}\n")
        f.write(f"Processing time: {processing_time/60:.2f} minutes\n\n")
        f.write(f"Category statistics:\n")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {category}: {count} keywords\n")
    
    # If there are uncategorized keywords, combine them
    if uncategorized_dfs:
        combined_uncategorized = pd.concat(uncategorized_dfs, ignore_index=True)
        
        # Remove duplicates
        combined_uncategorized.drop_duplicates(subset=['Keyword'], keep='first', inplace=True)
        
        logger.info(f"Combined {len(combined_uncategorized)} unique uncategorized keywords for ML processing")
        
        # Save uncategorized keywords for debugging or manual review
        uncategorized_file = os.path.join(ENHANCED_DIR, "uncategorized.csv")
        combined_uncategorized.to_csv(uncategorized_file, index=False)
        
        return combined_uncategorized
    else:
        logger.info("No uncategorized keywords found after fast categorization")
        return None

def categorize_with_patterns(keyword):
    """
    Use expanded patterns for categorization.
    This is called for keywords that didn't match any categories in the fast categorization step.
    """
    if not isinstance(keyword, str) or not keyword.strip():
        return []
    
    categories = set()
    
    # Check for expanded categories
    for category_name, patterns in CATEGORY_PATTERNS.items():
        if patterns.search(keyword):
            categories.add(category_name)
    
    # Special cases for ethnic terms
    if any(re.search(pattern, keyword, re.IGNORECASE) for pattern in EXPANDED_ETHNICITIES):
        categories.add('ethnicities')
    
    # Website terms
    if any(re.search(pattern, keyword, re.IGNORECASE) for pattern in EXPANDED_WEBSITE_TERMS):
        categories.add('website_terms')
        
    # Additional category terms
    if any(re.search(pattern, keyword, re.IGNORECASE) for pattern in EXPANDED_CATEGORY):
        categories.add('category3')
    
    # Check for person names
    if is_likely_person_name(keyword):
        categories.add('person_names')
    
    # Special rules based on word patterns
    
    # Check for website domain pattern
    if re.search(r'[a-zA-Z0-9\-]+\.(com|org|net|co|io|info|biz|xyz)', keyword, re.IGNORECASE):
        categories.add('website_terms')
    
    # Check for online terms
    if re.search(r'\bonline\b|\bon[-\s]?line\b', keyword, re.IGNORECASE):
        categories.add('website_terms')
    
    # Check for question patterns (how, what, when)
    if re.search(r'^(how|what|when|why|where|who)\s', keyword, re.IGNORECASE):
        categories.add('review')
    
    # Rule for role_play categories
    if re.search(r'\b(role_term1|role_term2|role_term3|role_term4)\b', keyword, re.IGNORECASE):
        categories.add('role_play')
    
    # Add URL pattern check
    if URL_PATTERN.search(keyword):
        categories.add('website_terms')
    
    # If keyword contains emoji
    if EMOJI_PATTERN.search(keyword):
        categories.add('emoji')
    
    # Check language patterns
    language = detect_language(keyword)
    if language:
        categories.add(language)
    
    # If no categories found, add misc
    if not categories:
        categories.add('misc')
    
    # Handle specific rules
    
    # Rule 30: ALL keywords with "programa" should be in service_category and spanish
    if re.search(r'\bprograma\b', keyword, re.IGNORECASE):
        categories.add('spanish')
        categories.add('service_category')
    
    # Remove Vietnamese from east_european
    if 'vietnamese' in categories and 'east_european' in categories:
        categories.remove('east_european')
    
    return list(categories)

def is_likely_person_name(text):
    """Check if a text string is likely to be a person's name"""
    if not isinstance(text, str) or not text.strip():
        return False
        
    # Split by common separators
    parts = re.split(r'[\s\-\_\.\+]+', text.lower())
    
    # Filter out empty parts
    parts = [p for p in parts if p]
    
    if not parts:
        return False
    
    # Very short text (single character) is not a name
    if all(len(p) <= 1 for p in parts):
        return False
    
    # Check if text contains digits or special characters (not names)
    if re.search(r'\d|[^\w\s\-\.\']', text):
        return False
        
    # Check if any part matches known first or last names
    matches = sum(1 for part in parts if part in COMMON_FIRST_NAMES or part in COMMON_LAST_NAMES)
    
    # If there's at least one match and not too many parts, it's likely a name
    if matches > 0 and len(parts) <= 4:
        return True
        
    # Names typically have 2-4 parts, check for length and capitalization patterns
    if 1 <= len(parts) <= 4:
        # Check if all parts start with uppercase in the original text
        original_parts = re.split(r'[\s\-\_\.\+]+', text)
        original_parts = [p for p in original_parts if p]
        
        if all(p and p[0].isupper() for p in original_parts):
            # Names should have reasonable lengths
            if all(2 <= len(p) <= 12 for p in parts):
                # Rule out common non-name terms
                non_name_terms = {'video', 'category1', 'category2', 'category3', 'category4', 'category5', 'site1', 'site2', 'term1', 'term2'}
                if not any(part in non_name_terms for part in parts):
                    return True
    
    return False

def is_non_english_language(keyword):
    """
    Check if a keyword is likely in a non-English language
    based on character patterns and common words
    """
    if not isinstance(keyword, str) or not keyword.strip():
        return False
        
    # Check for Japanese characters
    if JAPANESE_PATTERN.search(keyword):
        return True
        
    # Check for Korean characters
    if KOREAN_PATTERN.search(keyword):
        return True
        
    # Check for Chinese characters
    if CHINESE_PATTERN.search(keyword):
        return True
        
    # Check for Thai characters
    if THAI_PATTERN.search(keyword):
        return True
        
    # Check for Indian script characters
    if INDIAN_PATTERN.search(keyword):
        return True
        
    # Check for Turkish characters
    if TURKISH_PATTERN.search(keyword):
        return True
        
    # Check for Middle Eastern script characters
    if MIDDLE_EASTERN_PATTERN.search(keyword):
        return True
        
    # Check for Russian/Cyrillic characters
    if RUSSIAN_PATTERN.search(keyword):
        return True
        
    # Check for Spanish unique characters or common Spanish words
    if SPANISH_PATTERN.search(keyword):
        return True

    # Check for Portuguese unique characters or common Portuguese words
    if PORTUGUESE_PATTERN.search(keyword):
        return True
        
    # Check for Vietnamese patterns
    if any(re.search(pattern, keyword, re.IGNORECASE) for pattern in EXPANDED_VIETNAMESE):
        return True
    
    # Check for East European characters
    if EAST_EUROPEAN_PATTERN.search(keyword):
        return True
        
    return False

def train_ml_model(uncategorized_df):
    """
    Train ML models using the already categorized keywords to 
    categorize the remaining uncategorized keywords
    """
    # Use categorized keywords from the output directory as training data
    all_categorized_data = []
    category_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    
    if not category_files:
        logger.warning("No category files found for ML training")
        return None, None, None
    
    all_categories = {}
    # Read each category file
    for category_file in category_files:
        category_name = os.path.basename(category_file).replace('.csv', '')
        
        # Skip reading person_names for training due to size and relevance
        if category_name == 'person_names':
            continue
            
        try:
            category_df = pd.read_csv(category_file)
            if 'Keyword' not in category_df.columns:
                logger.warning(f"Invalid format in {category_file}, skipping")
                continue
                
            # Add category as a column
            category_df['ML_Category'] = category_name
            
            # Keep track of category sizes
            all_categories[category_name] = len(category_df)
            
            # Add to combined training data
            all_categorized_data.append(category_df[['Keyword', 'ML_Category']])
            
        except Exception as e:
            logger.error(f"Error reading {category_file}: {e}")
    
    if not all_categorized_data:
        logger.warning("No valid category data found for ML training")
        return None, None, None
        
    # Combine all categorized data
    training_df = pd.concat(all_categorized_data, ignore_index=True)
    
    # Remove duplicates, keeping first occurrence (in case a keyword is in multiple categories)
    training_df.drop_duplicates(subset=['Keyword'], keep='first', inplace=True)
    
    logger.info(f"Training ML model with {len(training_df)} keywords and {len(all_categories)} categories")
    logger.info(f"Category distribution: {all_categories}")
    
    # Split data into training and test sets
    X = training_df['Keyword']
    y = training_df['ML_Category']
    
    # Check if we have enough data for splitting
    if len(X) < 100:
        logger.warning(f"Only {len(X)} training examples, which is not enough for ML. Using simple pattern matching.")
        return None, None, list(all_categories.keys())
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create TF-IDF vectorizer for keyword features
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=2, max_df=0.9, sublinear_tf=True)
        
        # Try both Random Forest and Naive Bayes classifiers
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        nb_classifier = MultinomialNB()
        
        # Define pipelines
        rf_pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', rf_classifier)
        ])
        
        nb_pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', nb_classifier)
        ])
        
        # Train both pipelines
        rf_pipeline.fit(X_train, y_train)
        nb_pipeline.fit(X_train, y_train)
        
        # Test on the test set
        rf_score = rf_pipeline.score(X_test, y_test)
        nb_score = nb_pipeline.score(X_test, y_test)
        
        # Choose the better classifier
        if rf_score >= nb_score:
            logger.info(f"Using Random Forest classifier (accuracy: {rf_score:.4f})")
            classifier = rf_pipeline
        else:
            logger.info(f"Using Naive Bayes classifier (accuracy: {nb_score:.4f})")
            classifier = nb_pipeline
        
        # Print classification report for the best model
        y_pred = classifier.predict(X_test)
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        
        # Save the model and vectorizer
        joblib.dump(classifier, os.path.join(ENHANCED_DIR, "keyword_classifier.pkl"))
        
        # Train a special classifier just for person name detection
        name_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        name_classifier = None
        
        # Try to get person_names data for a binary classifier
        person_names_file = os.path.join(OUTPUT_DIR, "person_names.csv")
        if os.path.exists(person_names_file):
            try:
                names_df = pd.read_csv(person_names_file)
                if 'Keyword' in names_df.columns and len(names_df) > 50:  # Need enough examples
                    # Create positive examples (names)
                    names_df = names_df.sample(min(len(names_df), 5000), random_state=42)  # Limit to 5000 names
                    names_df['is_name'] = 1
                    
                    # Create negative examples (non-names) from other categories
                    non_names_df = training_df.sample(min(len(names_df), len(training_df)), random_state=42)
                    non_names_df['is_name'] = 0
                    
                    # Combine and shuffle
                    name_training_df = pd.concat([names_df[['Keyword', 'is_name']], non_names_df[['Keyword', 'is_name']]], ignore_index=True)
                    name_training_df = name_training_df.sample(frac=1, random_state=42)
                    
                    # Train name classifier
                    X_name = name_training_df['Keyword']
                    y_name = name_training_df['is_name']
                    
                    X_name_train, X_name_test, y_name_train, y_name_test = train_test_split(X_name, y_name, test_size=0.2, random_state=42)
                    
                    # Use Random Forest for name classification
                    name_classifier = Pipeline([
                        ('tfidf', name_vectorizer),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                    ])
                    
                    name_classifier.fit(X_name_train, y_name_train)
                    name_score = name_classifier.score(X_name_test, y_name_test)
                    
                    logger.info(f"Person name classifier trained with accuracy: {name_score:.4f}")
                    joblib.dump(name_classifier, os.path.join(ENHANCED_DIR, "name_classifier.pkl"))
            except Exception as e:
                logger.error(f"Error training name classifier: {e}")
                name_classifier = None
        
        return vectorizer, classifier, list(all_categories.keys()), name_classifier
        
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        return None, None, list(all_categories.keys())

def extract_name_features(text):
    """Extract features from text that might indicate it is a person name"""
    if not isinstance(text, str) or not text.strip():
        return {}
    
    # Clean the text
    text = text.strip()
    
    # Split by common separators
    parts = re.split(r'[\s\-\_\.\+]+', text.lower())
    parts = [p for p in parts if p]
    
    features = {
        'num_parts': len(parts),
        'avg_part_len': sum(len(p) for p in parts) / max(1, len(parts)),
        'has_digit': int(bool(re.search(r'\d', text))),
        'has_special': int(bool(re.search(r'[^\w\s\-\.\']', text))),
        'all_parts_capitalize': int(all(p and p[0].isupper() for p in re.split(r'[\s\-\_\.\+]+', text) if p)),
        'first_part_capitalize': int(bool(text and text[0].isupper())),
        'common_name_match': sum(1 for part in parts if part in COMMON_FIRST_NAMES or part in COMMON_LAST_NAMES),
    }
    
    # Check for name patterns
    if 1 <= len(parts) <= 4 and not features['has_digit'] and not features['has_special']:
        # Common name patterns
        # FirstName LastName
        if len(parts) == 2 and features['all_parts_capitalize']:
            features['likely_full_name'] = 1
        # FirstName MiddleName LastName
        elif len(parts) == 3 and features['all_parts_capitalize']:
            features['likely_full_name_middle'] = 1
        # One word capitalized name
        elif len(parts) == 1 and features['first_part_capitalize'] and 3 <= len(parts[0]) <= 10:
            features['likely_single_name'] = 1
    else:
        features['likely_full_name'] = 0
        features['likely_full_name_middle'] = 0
        features['likely_single_name'] = 0
    
    return features

def categorize_with_ml(uncategorized_df, vectorizer, classifier, category_names, name_classifier=None):
    """Use ML model to categorize uncategorized keywords"""
    if uncategorized_df is None or len(uncategorized_df) == 0:
        logger.info("No uncategorized keywords to process with ML")
        return {}
    
    # Create a copy to avoid modifying the original
    df = uncategorized_df.copy()
    
    # If vectorizer and classifier are not available, use pattern matching
    if vectorizer is None or classifier is None:
        logger.info("ML model not available. Using pattern matching for categorization.")
        df['ML_Categories'] = df['Keyword'].apply(categorize_with_patterns)
    else:
        # Use the ML model for prediction
        try:
            # Get the classifier's predicted category
            predicted_categories = classifier.predict(df['Keyword'])
            df['ML_Category'] = predicted_categories
            
            # Get prediction probabilities to estimate confidence
            probabilities = classifier.predict_proba(df['Keyword'])
            df['Confidence'] = [max(probs) for probs in probabilities]
            
            # For low confidence predictions, add pattern-based categories
            df['ML_Categories'] = df.apply(
                lambda row: ([row['ML_Category']] if row['Confidence'] > 0.7 else 
                          categorize_with_patterns(row['Keyword'])), 
                axis=1
            )
            
            # If name classifier is available, check for person names
            if name_classifier is not None:
                # Predict if each keyword is a person name
                name_probs = name_classifier.predict_proba(df['Keyword'])
                df['name_probability'] = [probs[1] for probs in name_probs]  # Probability of being a name
                
                # Add person_names category for high-probability names
                df['ML_Categories'] = df.apply(
                    lambda row: (row['ML_Categories'] + ['person_names'] 
                              if row['name_probability'] > 0.8 and 'person_names' not in row['ML_Categories']
                              else row['ML_Categories']),
                    axis=1
                )
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            df['ML_Categories'] = df['Keyword'].apply(categorize_with_patterns)
    
    # Apply results to store keywords in respective category files
    category_counts = {}
    
    # Process each uncategorized keyword
    for _, row in df.iterrows():
        keyword = row['Keyword']
        categories = row['ML_Categories']
        
        if not categories:  # Skip if no categories assigned
            continue
            
        # Update category counts
        for category in categories:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        # Add to each category file
        for category in categories:
            category_file = os.path.join(OUTPUT_DIR, f"{category}.csv")
            
            row_data = {
                'Keyword': keyword,
                'Volume': row.get('Volume', 1),
                'Source': row.get('Source', 'ML_categorization'),
                'Categories': str(categories)
            }
            
            try:
                # Create a one-row dataframe
                row_df = pd.DataFrame([row_data])
                
                # Check if file exists
                if os.path.exists(category_file):
                    # Append to file
                    row_df.to_csv(category_file, mode='a', header=False, index=False)
                else:
                    # Create new file
                    row_df.to_csv(category_file, index=False)
            except Exception as e:
                logger.error(f"Error writing {keyword} to {category}.csv: {e}")
    
    return category_counts

def step2_ml_categorization(uncategorized_df):
    """
    Step 2: ML-based categorization for keywords that weren't categorized in step 1
    """
    if uncategorized_df is None or len(uncategorized_df) == 0:
        logger.info("No uncategorized keywords to process with ML")
        return {}
        
    logger.info(f"Starting ML categorization for {len(uncategorized_df)} uncategorized keywords")
    
    start_time = time.time()
    
    # Train ML model
    vectorizer, classifier, category_names, name_classifier = train_ml_model(uncategorized_df)
    
    # Apply ML model to categorize keywords
    category_counts = categorize_with_ml(uncategorized_df, vectorizer, classifier, category_names, name_classifier)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) / 60
    logger.info(f"ML categorization complete in {processing_time:.2f} minutes")
    
    # Display category distribution
    if category_counts:
        total_categorized = sum(category_counts.values())
        logger.info(f"ML categorized {total_categorized} keywords across {len(category_counts)} categories")
        
        # Sort categories by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Display top categories
        logger.info("Top 10 categories from ML categorization:")
        for category, count in sorted_categories[:10]:
            logger.info(f"  {category}: {count} keywords")
            
        # Save results to a summary file
        with open(os.path.join(ENHANCED_DIR, "ml_categorization_summary.txt"), "w", encoding='utf-8') as f:
            f.write(f"ML Categorization Summary\n")
            f.write(f"=========================\n\n")
            f.write(f"Total keywords categorized: {total_categorized}\n")
            f.write(f"Total categories: {len(category_counts)}\n\n")
            f.write(f"Category distribution:\n")
            for category, count in sorted_categories:
                f.write(f"  {category}: {count} keywords\n")
    else:
        logger.info("No keywords were categorized by ML")
    
    return category_counts

def cross_verify_person_names():
    """
    Cross-verify potential person names by looking for patterns across 
    different category files and compile a comprehensive person_names.csv
    """
    logger.info("Starting cross-verification of person names")
    
    # Get all category files
    category_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if not category_files:
        logger.warning("No category files found for person name verification")
        return 0
    
    # Load existing person_names if available
    person_names_file = os.path.join(OUTPUT_DIR, "person_names.csv")
    existing_person_names = set()
    
    if os.path.exists(person_names_file):
        try:
            person_df = pd.read_csv(person_names_file)
            if 'Keyword' in person_df.columns:
                existing_person_names = set(person_df['Keyword'].astype(str))
                logger.info(f"Loaded {len(existing_person_names)} existing person names")
        except Exception as e:
            logger.error(f"Error reading person_names.csv: {e}")
    
    # Process each category file for potential person names
    potential_names = []
    name_sources = {}
    processed_keywords = set()
    
    for category_file in category_files:
        category_name = os.path.basename(category_file).replace('.csv', '')
        
        # Skip person_names itself
        if category_name == 'person_names':
            continue
        
        try:
            category_df = pd.read_csv(category_file)
            if 'Keyword' not in category_df.columns:
                continue
                
            # Check each keyword
            for _, row in category_df.iterrows():
                keyword = str(row['Keyword']).strip()
                
                # Skip already processed keywords
                if keyword in processed_keywords or keyword in existing_person_names:
                    continue
                    
                processed_keywords.add(keyword)
                
                # Check if it's a potential person name
                if is_likely_person_name(keyword):
                    # Keep track of categories this name appears in
                    if keyword in name_sources:
                        name_sources[keyword].add(category_name)
                    else:
                        name_sources[keyword] = {category_name}
                        
                    # Add to potential names if not already added
                    if keyword not in [item['Keyword'] for item in potential_names]:
                        new_row = {
                            'Keyword': keyword,
                            'Volume': row.get('Volume', 1),
                            'Source': f"cross_verify:{row.get('Source', category_name)}",
                            'Categories': str(['person_names'])
                        }
                        potential_names.append(new_row)
            
        except Exception as e:
            logger.error(f"Error processing {category_file} for person names: {e}")
    
    # Enhance confidence by preferring names that appear in multiple categories
    confident_names = []
    
    for item in potential_names:
        keyword = item['Keyword']
        sources = name_sources.get(keyword, set())
        
        # Higher confidence if name appears in multiple categories or in typical categories
        typical_name_categories = {'content_interest', 'public_figure', 'category1', 'category2'}
        high_confidence = (
            len(sources) > 1 or  # Appears in multiple categories
            any(category in sources for category in typical_name_categories)  # Appears in typical categories
        )
        
        if high_confidence:
            confident_names.append(item)
    
    # If too few confident names, add all potential names
    if len(confident_names) < 10 and len(potential_names) > 10:
        logger.info(f"Only {len(confident_names)} confident names found, using all {len(potential_names)} potential names")
        confident_names = potential_names
    
    # Add new names to person_names.csv
    if confident_names:
        new_names_df = pd.DataFrame(confident_names)
        
        # Create or append to person_names.csv
        if os.path.exists(person_names_file):
            new_names_df.to_csv(person_names_file, mode='a', header=False, index=False)
        else:
            new_names_df.to_csv(person_names_file, index=False)
        
        logger.info(f"Added {len(confident_names)} new person names from cross-verification")
        return len(confident_names)
    else:
        logger.info("No new person names found in cross-verification")
        return 0

def process_person_name_files():
    """
    Process special Excel files containing person names and ensure 
    they're properly categorized
    """
    logger.info("Processing special Excel files containing person names")
    
    all_person_data = []
    processed_count = 0
    
    # Process each special Excel file
    for file_name in PERSON_NAME_FILES:
        file_path = os.path.join(PERSON_SOURCES_DIR, file_name)
        
        if not os.path.exists(file_path):
            logger.warning(f"Special person name file not found: {file_path}")
            continue
            
        logger.info(f"Processing special person name file: {file_name}")
        
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Standardize column names
            if len(df.columns) > 0:
                df.rename(columns={df.columns[0]: 'Keyword'}, inplace=True)
                
                # Ensure Volume column exists
                if 'Volume' not in df.columns and len(df.columns) > 1:
                    df.rename(columns={df.columns[1]: 'Volume'}, inplace=True)
                if 'Volume' not in df.columns:
                    df['Volume'] = 1
                
                # Add source column
                df['Source'] = file_name
                
                # Clean up keyword column
                df['Keyword'] = df['Keyword'].astype(str).str.strip()
                
                # Assign appropriate categories based on file source
                df['Categories'] = [PERSON_FILE_CATEGORIES.get(file_name, ["person_names"]) for _ in range(len(df))]
                
                # Add to overall person data
                all_person_data.append(df)
                processed_count += len(df)
                
                logger.info(f"Processed {len(df)} names from {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    if not all_person_data:
        logger.warning("No person name data found in special Excel files")
        return 0
        
    # Combine all person data
    combined_person_df = pd.concat(all_person_data, ignore_index=True)
    logger.info(f"Combined {len(combined_person_df)} person names from special files")
    
    # Remove duplicates (keeping first occurrence)
    combined_person_df.drop_duplicates(subset=['Keyword'], keep='first', inplace=True)
    logger.info(f"After removing duplicates: {len(combined_person_df)} unique person names")
    
    # Load existing person_names.csv if it exists
    person_names_file = os.path.join(OUTPUT_DIR, "person_names.csv")
    existing_names = set()
    
    if os.path.exists(person_names_file):
        try:
            existing_df = pd.read_csv(person_names_file)
            existing_names = set(existing_df['Keyword'].astype(str))
            logger.info(f"Found {len(existing_names)} existing entries in person_names.csv")
        except Exception as e:
            logger.error(f"Error reading existing person_names.csv: {e}")
    
    # Filter to only new names
    new_names_df = combined_person_df[~combined_person_df['Keyword'].isin(existing_names)]
    logger.info(f"Adding {len(new_names_df)} new person names to person_names.csv")
    
    # Add new names to person_names.csv
    if len(new_names_df) > 0:
        if os.path.exists(person_names_file):
            new_names_df.to_csv(person_names_file, mode='a', header=False, index=False)
        else:
            new_names_df.to_csv(person_names_file, index=False)
    
    # Ensure names are added to their respective category files
    for idx, row in combined_person_df.iterrows():
        # For each category this name belongs to
        for category in row['Categories']:
            if category == 'person_names':
                continue  # Already handled above
                
            category_file = os.path.join(OUTPUT_DIR, f"{category}.csv")
            
            try:
                # Create a one-row dataframe
                row_data = {
                    'Keyword': row['Keyword'],
                    'Volume': row['Volume'],
                    'Source': row['Source'],
                    'Categories': row['Categories']
                }
                new_row_df = pd.DataFrame([row_data])
                
                # Check if file exists
                if os.path.exists(category_file):
                    # Check if name already exists in this category
                    try:
                        cat_df = pd.read_csv(category_file)
                        if row['Keyword'] in cat_df['Keyword'].values:
                            continue  # Skip if already in category file
                    except:
                        pass  # If error reading, try to append anyway
                        
                    # Append to file
                    new_row_df.to_csv(category_file, mode='a', header=False, index=False)
                else:
                    # Create new file
                    new_row_df.to_csv(category_file, index=False)
            except Exception as e:
                logger.error(f"Error adding {row['Keyword']} to {category}.csv: {e}")
    
    logger.info(f"Completed processing of special person name files")
    return len(new_names_df)

def extract_all_chinese_keywords():
    """
    Extract all keywords containing Chinese characters from all category files
    and create a dedicated Chinese keywords CSV file.
    """
    logger.info("Creating dedicated CSV file for all Chinese keywords")
    
    # Find all CSV files in the output directory
    all_csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    
    # Create a set to store unique Chinese keywords
    chinese_keywords = set()
    chinese_keyword_data = []
    
    # Process each CSV file to find Chinese keywords
    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if 'Keyword' not in df.columns:
                continue
                
            # Extract keywords with Chinese characters
            for _, row in df.iterrows():
                keyword = str(row['Keyword'])
                # Check for Chinese characters
                if CHINESE_PATTERN.search(keyword):
                    if keyword not in chinese_keywords:
                        chinese_keywords.add(keyword)
                        
                        # Prepare data for the Chinese keywords CSV
                        row_data = {
                            'Keyword': keyword,
                            'Volume': row.get('Volume', 50),  # Default to 50 if not present
                            'Source': row.get('Source', os.path.basename(csv_file)),
                            'Categories': ['chinese'] if 'Categories' not in row else row['Categories'] + ['chinese']
                        }
                        chinese_keyword_data.append(row_data)
        except Exception as e:
            logger.error(f"Error processing {csv_file} for Chinese keywords: {e}")
    
    # Create the Chinese keywords CSV file
    if chinese_keyword_data:
        chinese_file = os.path.join(OUTPUT_DIR, "chinese.csv")
        chinese_df = pd.DataFrame(chinese_keyword_data)
        chinese_df.to_csv(chinese_file, index=False, encoding='utf-8')
        logger.info(f"Saved {len(chinese_keyword_data)} Chinese keywords to chinese.csv")
    else:
        logger.info("No Chinese keywords found.")
    
    return len(chinese_keyword_data)

def main():
    """Main function to run the combined categorization process"""
    total_start_time = time.time()
    logger.info("Starting combined keyword categorization (fast + ML)")
    
    # Step 1: Fast categorization
    uncategorized_df = step1_fast_categorization()
    
    # Step 1.5: Process person name files (new step)
    num_new_names_files = process_person_name_files()
    logger.info(f"Added {num_new_names_files} new names from special Excel files")
    
    if uncategorized_df is None or len(uncategorized_df) == 0:
        logger.info("No uncategorized keywords found after fast categorization. Skipping ML step.")
        # Still proceed with other steps
    else:
        # Step 2: ML categorization for remaining uncategorized keywords
        category_counts = step2_ml_categorization(uncategorized_df)
    
    # Step 3: Cross-verify person names across all categories
    num_new_names = cross_verify_person_names()
    
    # Step 4: Extract all Chinese keywords to a dedicated CSV
    num_chinese_keywords = extract_all_chinese_keywords()
    
    # Generate overall summary
    total_time = (time.time() - total_start_time) / 60
    logger.info(f"Combined categorization complete! Total processing time: {total_time:.2f} minutes")
    
    with open(os.path.join(ENHANCED_DIR, "combined_categorization_summary.txt"), "w", encoding='utf-8') as f:
        f.write(f"Combined Keyword Categorization Summary\n")
        f.write(f"====================================\n\n")
        f.write(f"Total processing time: {total_time:.2f} minutes\n\n")
        f.write(f"Workflow:\n")
        f.write(f"1. Fast categorization of all keywords\n")
        f.write(f"1.5. Processed special Excel files for person names ({num_new_names_files} new names added)\n")
        f.write(f"2. ML categorization of remaining uncategorized keywords\n")
        f.write(f"3. Cross-verification of person names across categories ({num_new_names} new names added)\n")
        f.write(f"4. Extracted all Chinese keywords to a dedicated CSV ({num_chinese_keywords} keywords)\n\n")
        f.write(f"Results are saved to {OUTPUT_DIR}/ directory\n")

if __name__ == "__main__":
    main() 