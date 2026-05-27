"""
📝 Text Data Processor
Specialized processor for text data with NLP features
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text Data Processor
    
    Specialized processor for text data with intelligent:
    - Text preprocessing and cleaning
    - Feature extraction (TF-IDF, n-grams, embeddings)
    - Text statistics and metadata
    - Language detection
    - Sentiment analysis
    """
    
    def __init__(self):
        self.vectorizers = {}
        self.scalers = {}
        self.text_stats = {}
        self.processing_history = []
        self.vocabulary = {}
        
    def process(self, X: pd.DataFrame, y: pd.Series = None, 
                config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process text data with intelligent NLP feature engineering
        
        Args:
            X: Feature data (should include text columns)
            y: Target data (optional)
            config: Processing configuration
            
        Returns:
            Processed data and processing metadata
        """
        logger.info("📝 Processing text data")
        start_time = time.time()
        
        config = config or self._get_default_config()
        metadata = {
            "original_shape": X.shape,
            "processing_steps": [],
            "text_columns": [],
            "text_stats": {}
        }
        
        X_processed = X.copy()
        
        # Step 1: Detect text columns
        text_columns = self._detect_text_columns(X_processed)
        metadata["text_columns"] = text_columns
        
        if not text_columns:
            logger.warning("No text columns detected")
            return X_processed, metadata
        
        # Step 2: Clean and preprocess text
        if config.get("clean_text", True):
            X_processed, cleaning_metadata = self._clean_text_data(
                X_processed, text_columns, config
            )
            metadata["text_cleaning"] = cleaning_metadata
            metadata["processing_steps"].append("text_cleaning")
        
        # Step 3: Extract text statistics
        if config.get("extract_text_stats", True):
            X_processed, stats_metadata = self._extract_text_statistics(
                X_processed, text_columns, config
            )
            metadata["text_statistics"] = stats_metadata
            metadata["processing_steps"].append("text_statistics")
        
        # Step 4: Create n-gram features
        if config.get("create_ngrams", True):
            X_processed, ngram_metadata = self._create_ngram_features(
                X_processed, text_columns, config
            )
            metadata["ngram_features"] = ngram_metadata
            metadata["processing_steps"].append("ngrams")
        
        # Step 5: Create TF-IDF features
        if config.get("create_tfidf", True):
            X_processed, tfidf_metadata = self._create_tfidf_features(
                X_processed, text_columns, config
            )
            metadata["tfidf_features"] = tfidf_metadata
            metadata["processing_steps"].append("tfidf")
        
        # Step 6: Create sentiment features
        if config.get("create_sentiment", True):
            X_processed, sentiment_metadata = self._create_sentiment_features(
                X_processed, text_columns, config
            )
            metadata["sentiment_features"] = sentiment_metadata
            metadata["processing_steps"].append("sentiment")
        
        # Step 7: Create linguistic features
        if config.get("create_linguistic", True):
            X_processed, linguistic_metadata = self._create_linguistic_features(
                X_processed, text_columns, config
            )
            metadata["linguistic_features"] = linguistic_metadata
            metadata["processing_steps"].append("linguistic")
        
        # Step 8: Create readability features
        if config.get("create_readability", True):
            X_processed, readability_metadata = self._create_readability_features(
                X_processed, text_columns, config
            )
            metadata["readability_features"] = readability_metadata
            metadata["processing_steps"].append("readability")
        
        # Step 9: Handle missing values
        if config.get("handle_missing", True):
            X_processed, missing_metadata = self._handle_text_missing(
                X_processed, config
            )
            metadata["missing_handling"] = missing_metadata
            metadata["processing_steps"].append("missing_values")
        
        # Step 10: Scale features
        if config.get("scale_features", True):
            X_processed, scaling_metadata = self._scale_text_features(
                X_processed, config
            )
            metadata["feature_scaling"] = scaling_metadata
            metadata["processing_steps"].append("feature_scaling")
        
        # Step 11: Drop original text columns (optional)
        if config.get("drop_text_columns", True):
            X_processed = X_processed.drop(columns=text_columns)
        
        processing_time = time.time() - start_time
        
        metadata.update({
            "final_shape": X_processed.shape,
            "processing_time": processing_time,
            "features_created": X_processed.shape[1] - X.shape[1]
        })
        
        # Store processing history
        self.processing_history.append(metadata)
        
        logger.info(f"✅ Text processing complete: {X.shape} → {X_processed.shape} in {processing_time:.2f}s")
        return X_processed, metadata
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default text processing configuration"""
        return {
            "clean_text": True,
            "extract_text_stats": True,
            "create_ngrams": True,
            "create_tfidf": True,
            "create_sentiment": True,
            "create_linguistic": True,
            "create_readability": True,
            "handle_missing": True,
            "scale_features": True,
            "drop_text_columns": True,
            
            # Text cleaning parameters
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": False,
            "remove_stopwords": True,
            "min_word_length": 2,
            "max_word_length": 20,
            
            # N-gram parameters
            "ngram_range": (1, 2),
            "max_ngram_features": 1000,
            "min_ngram_freq": 2,
            
            # TF-IDF parameters
            "max_tfidf_features": 1000,
            "min_tfidf_freq": 2,
            "tfidf_norm": "l2",
            
            # Sentiment parameters
            "sentiment_lexicon": "vader",  # vader, textblob, custom
            
            # Scaling parameters
            "scaling_method": "standard"  # standard, minmax, robust
        }
    
    def _detect_text_columns(self, X: pd.DataFrame) -> List[str]:
        """Detect text columns in the dataset"""
        text_columns = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                # Check if it's actually text (not categorical with few values)
                sample_size = min(100, len(X[col]))
                sample_texts = X[col].dropna().head(sample_size)
                
                if len(sample_texts) > 0:
                    # Calculate average text length
                    avg_length = sample_texts.str.len().mean()
                    
                    # Check if it's likely text (average length > 20 chars)
                    if avg_length > 20:
                        # Check if it contains mostly words (not just codes/IDs)
                        word_ratio = sample_texts.str.contains(r'\b\w+\b').mean()
                        if word_ratio > 0.5:
                            text_columns.append(col)
        
        return text_columns
    
    def _clean_text_data(self, X: pd.DataFrame, text_columns: List[str],
                        config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean and preprocess text data"""
        logger.info("🧹 Cleaning text data")
        
        cleaning_stats = {}
        
        for col in text_columns:
            original_texts = X[col].copy()
            cleaned_texts = X[col].copy()
            
            # Convert to string
            cleaned_texts = cleaned_texts.astype(str)
            
            # Lowercase
            if config.get("lowercase", True):
                cleaned_texts = cleaned_texts.str.lower()
            
            # Remove punctuation
            if config.get("remove_punctuation", True):
                cleaned_texts = cleaned_texts.str.replace(r'[^\w\s]', ' ', regex=True)
            
            # Remove numbers
            if config.get("remove_numbers", False):
                cleaned_texts = cleaned_texts.str.replace(r'\d+', ' ', regex=True)
            
            # Remove extra whitespace
            cleaned_texts = cleaned_texts.str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Remove stopwords (simplified)
            if config.get("remove_stopwords", True):
                stopwords = self._get_stopwords()
                cleaned_texts = cleaned_texts.apply(
                    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
                )
            
            # Filter by word length
            min_len = config.get("min_word_length", 2)
            max_len = config.get("max_word_length", 20)
            cleaned_texts = cleaned_texts.apply(
                lambda x: ' '.join([word for word in x.split() if min_len <= len(word) <= max_len])
            )
            
            # Update dataframe
            X[col] = cleaned_texts
            
            # Calculate cleaning statistics
            cleaning_stats[col] = {
                "original_avg_length": original_texts.str.len().mean(),
                "cleaned_avg_length": cleaned_texts.str.len().mean(),
                "length_reduction": (original_texts.str.len().mean() - cleaned_texts.str.len().mean()) / original_texts.str.len().mean(),
                "original_unique_words": len(set(' '.join(original_texts.dropna()).split())),
                "cleaned_unique_words": len(set(' '.join(cleaned_texts.dropna()).split())),
                "vocabulary_reduction": 0  # Will be calculated below
            }
            
            if cleaning_stats[col]["original_unique_words"] > 0:
                reduction = (cleaning_stats[col]["original_unique_words"] - cleaning_stats[col]["cleaned_unique_words"]) / cleaning_stats[col]["original_unique_words"]
                cleaning_stats[col]["vocabulary_reduction"] = reduction
        
        metadata = {
            "cleaning_stats": cleaning_stats,
            "columns_processed": text_columns
        }
        
        return X, metadata
    
    def _get_stopwords(self) -> set:
        """Get basic stopwords"""
        # Simplified stopwords - in practice, you'd use NLTK or spaCy
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'i', 'you', 'your', 'we', 'they', 'them', 'their', 'this', 'that', 'these',
            'those', 'or', 'but', 'not', 'no', 'yes', 'if', 'then', 'else', 'when', 'where',
            'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'can', 'could', 'should',
            'would', 'may', 'might', 'must', 'shall', 'do', 'does', 'did', 'have', 'had',
            'been', 'being', 'am', 'are', 'were', 'was', 'be', 'been', 'being'
        }
        return stopwords
    
    def _extract_text_statistics(self, X: pd.DataFrame, text_columns: List[str],
                               config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract text statistics features"""
        logger.info("📊 Extracting text statistics")
        
        stats_features = {}
        
        for col in text_columns:
            # Basic statistics
            stats_features[f'{col}_char_count'] = X[col].str.len()
            stats_features[f'{col}_word_count'] = X[col].str.split().str.len()
            stats_features[f'{col}_sentence_count'] = X[col].str.count(r'[.!?]+')
            stats_features[f'{col}_avg_word_length'] = X[col].apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
            )
            
            # Advanced statistics
            stats_features[f'{col}_unique_words'] = X[col].apply(
                lambda x: len(set(str(x).split())) if str(x).split() else 0
            )
            stats_features[f'{col}_lexical_diversity'] = (
                stats_features[f'{col}_unique_words'] / stats_features[f'{col}_word_count']
            ).fillna(0)
            
            # Punctuation statistics
            stats_features[f'{col}_exclamation_count'] = X[col].str.count('!')
            stats_features[f'{col}_question_count'] = X[col].str.count(r'\?')
            stats_features[f'{col}_comma_count'] = X[col].str.count(',')
            stats_features[f'{col}_period_count'] = X[col].str.count(r'\.')
            
            # Special character statistics
            stats_features[f'{col}_uppercase_count'] = X[col].str.count(r'[A-Z]')
            stats_features[f'{col}_digit_count'] = X[col].str.count(r'\d')
            stats_features[f'{col}_space_count'] = X[col].str.count(' ')
            
            # Readability indicators
            stats_features[f'{col}_words_per_sentence'] = (
                stats_features[f'{col}_word_count'] / stats_features[f'{col}_sentence_count']
            ).fillna(0)
        
        # Add features to dataframe
        for feature_name, feature_data in stats_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(stats_features.keys()),
            "statistics_calculated": [
                "char_count", "word_count", "sentence_count", "avg_word_length",
                "unique_words", "lexical_diversity", "punctuation_counts",
                "special_char_counts", "readability_indicators"
            ]
        }
        
        return X, metadata
    
    def _create_ngram_features(self, X: pd.DataFrame, text_columns: List[str],
                             config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create n-gram features"""
        logger.info("🔤 Creating n-gram features")
        
        ngram_range = config.get("ngram_range", (1, 2))
        max_features = config.get("max_ngram_features", 1000)
        min_freq = config.get("min_ngram_freq", 2)
        
        ngram_features = {}
        ngram_metadata = {}
        
        for col in text_columns:
            # Combine all texts for this column
            texts = X[col].fillna('').astype(str)
            
            # Create CountVectorizer for n-grams
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_freq,
                stop_words='english' if config.get("remove_stopwords", True) else None
            )
            
            try:
                # Fit and transform
                ngram_matrix = vectorizer.fit_transform(texts)
                
                # Convert to dataframe with meaningful column names
                feature_names = [f"{col}_ngram_{i}_{name}" for i, name in enumerate(vectorizer.get_feature_names_out())]
                ngram_df = pd.DataFrame(ngram_matrix.toarray(), columns=feature_names)
                
                # Add to main dataframe
                for feature_name in feature_names:
                    X[feature_name] = ngram_df[feature_name]
                    ngram_features[feature_name] = ngram_df[feature_name]
                
                # Store vectorizer
                self.vectorizers[f"{col}_ngram"] = vectorizer
                
                ngram_metadata[col] = {
                    "ngram_range": ngram_range,
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "features_created": len(feature_names)
                }
                
            except Exception as e:
                logger.warning(f"Failed to create n-grams for {col}: {e}")
                ngram_metadata[col] = {"error": str(e)}
        
        metadata = {
            "ngram_metadata": ngram_metadata,
            "features_created": list(ngram_features.keys())
        }
        
        return X, metadata
    
    def _create_tfidf_features(self, X: pd.DataFrame, text_columns: List[str],
                              config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create TF-IDF features"""
        logger.info("📝 Creating TF-IDF features")
        
        max_features = config.get("max_tfidf_features", 1000)
        min_freq = config.get("min_tfidf_freq", 2)
        norm = config.get("tfidf_norm", "l2")
        
        tfidf_features = {}
        tfidf_metadata = {}
        
        for col in text_columns:
            # Combine all texts for this column
            texts = X[col].fillna('').astype(str)
            
            # Create TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_freq,
                norm=norm,
                stop_words='english' if config.get("remove_stopwords", True) else None
            )
            
            try:
                # Fit and transform
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # Convert to dataframe with meaningful column names
                feature_names = [f"{col}_tfidf_{i}_{name}" for i, name in enumerate(vectorizer.get_feature_names_out())]
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
                
                # Add to main dataframe
                for feature_name in feature_names:
                    X[feature_name] = tfidf_df[feature_name]
                    tfidf_features[feature_name] = tfidf_df[feature_name]
                
                # Store vectorizer
                self.vectorizers[f"{col}_tfidf"] = vectorizer
                
                tfidf_metadata[col] = {
                    "max_features": max_features,
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "features_created": len(feature_names)
                }
                
            except Exception as e:
                logger.warning(f"Failed to create TF-IDF for {col}: {e}")
                tfidf_metadata[col] = {"error": str(e)}
        
        metadata = {
            "tfidf_metadata": tfidf_metadata,
            "features_created": list(tfidf_features.keys())
        }
        
        return X, metadata
    
    def _create_sentiment_features(self, X: pd.DataFrame, text_columns: List[str],
                                 config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create sentiment analysis features"""
        logger.info("😊 Creating sentiment features")
        
        sentiment_features = {}
        sentiment_metadata = {}
        
        for col in text_columns:
            try:
                # Simple sentiment analysis using word lists
                positive_words = self._get_positive_words()
                negative_words = self._get_negative_words()
                
                texts = X[col].fillna('').astype(str)
                
                # Calculate sentiment scores
                sentiment_scores = []
                for text in texts:
                    words = text.lower().split()
                    
                    positive_count = sum(1 for word in words if word in positive_words)
                    negative_count = sum(1 for word in words if word in negative_words)
                    total_words = len(words)
                    
                    if total_words > 0:
                        positive_ratio = positive_count / total_words
                        negative_ratio = negative_count / total_words
                        sentiment_score = (positive_count - negative_count) / total_words
                    else:
                        positive_ratio = negative_ratio = sentiment_score = 0
                    
                    sentiment_scores.append({
                        "positive_ratio": positive_ratio,
                        "negative_ratio": negative_ratio,
                        "sentiment_score": sentiment_score,
                        "positive_count": positive_count,
                        "negative_count": negative_count
                    })
                
                # Convert to dataframe columns
                sentiment_df = pd.DataFrame(sentiment_scores)
                sentiment_df.columns = [f"{col}_sentiment_{col_name}" for col_name in sentiment_df.columns]
                
                # Add to main dataframe
                for col_name in sentiment_df.columns:
                    X[col_name] = sentiment_df[col_name]
                    sentiment_features[col_name] = sentiment_df[col_name]
                
                sentiment_metadata[col] = {
                    "method": "word_list_based",
                    "positive_words_count": len(positive_words),
                    "negative_words_count": len(negative_words),
                    "features_created": len(sentiment_df.columns)
                }
                
            except Exception as e:
                logger.warning(f"Failed to create sentiment features for {col}: {e}")
                sentiment_metadata[col] = {"error": str(e)}
        
        metadata = {
            "sentiment_metadata": sentiment_metadata,
            "features_created": list(sentiment_features.keys())
        }
        
        return X, metadata
    
    def _get_positive_words(self) -> set:
        """Get positive sentiment words"""
        return {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'delighted',
            'perfect', 'best', 'brilliant', 'outstanding', 'superb', 'magnificent',
            'beautiful', 'nice', 'positive', 'favorable', 'beneficial', 'valuable'
        }
    
    def _get_negative_words(self) -> set:
        """Get negative sentiment words"""
        return {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'angry', 'sad', 'depressed', 'unhappy', 'disappointed', 'frustrated',
            'worst', 'poor', 'useless', 'worthless', 'pathetic', 'annoying',
            'ugly', 'negative', 'unfavorable', 'harmful', 'damaging', 'destructive'
        }
    
    def _create_linguistic_features(self, X: pd.DataFrame, text_columns: List[str],
                                  config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create linguistic features"""
        logger.info("🗣️ Creating linguistic features")
        
        linguistic_features = {}
        
        for col in text_columns:
            texts = X[col].fillna('').astype(str)
            
            # Part-of-speech features (simplified)
            linguistic_features[f'{col}_noun_ratio'] = texts.apply(self._estimate_noun_ratio)
            linguistic_features[f'{col}_verb_ratio'] = texts.apply(self._estimate_verb_ratio)
            linguistic_features[f'{col}_adj_ratio'] = texts.apply(self._estimate_adj_ratio)
            
            # Readability scores (simplified)
            linguistic_features[f'{col}_flesch_score'] = texts.apply(self._calculate_flesch_score)
            
            # Complexity features
            linguistic_features[f'{col}_avg_sentence_complexity'] = texts.apply(
                lambda x: self._calculate_sentence_complexity(str(x))
            )
        
        # Add features to dataframe
        for feature_name, feature_data in linguistic_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(linguistic_features.keys()),
            "linguistic_features": [
                "pos_ratios", "readability_scores", "complexity_measures"
            ]
        }
        
        return X, metadata
    
    def _estimate_noun_ratio(self, text: str) -> float:
        """Estimate noun ratio (simplified)"""
        # This is a very simplified version - in practice, you'd use NLTK or spaCy
        noun_indicators = ['tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ism']
        words = text.lower().split()
        
        if not words:
            return 0.0
        
        noun_count = sum(1 for word in words if any(indicator in word for indicator in noun_indicators))
        return noun_count / len(words)
    
    def _estimate_verb_ratio(self, text: str) -> float:
        """Estimate verb ratio (simplified)"""
        verb_indicators = ['ing', 'ed', 'es']
        words = text.lower().split()
        
        if not words:
            return 0.0
        
        verb_count = sum(1 for word in words if any(indicator in word for indicator in verb_indicators))
        return verb_count / len(words)
    
    def _estimate_adj_ratio(self, text: str) -> float:
        """Estimate adjective ratio (simplified)"""
        adj_indicators = ['ful', 'ous', 'ive', 'able', 'ible', 'al', 'ic', 'ish']
        words = text.lower().split()
        
        if not words:
            return 0.0
        
        adj_count = sum(1 for word in words if any(indicator in word for indicator in adj_indicators))
        return adj_count / len(words)
    
    def _calculate_flesch_score(self, text: str) -> float:
        """Calculate simplified Flesch reading ease score"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch formula
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, flesch_score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        vowels = 'aeiouy'
        word = word.lower()
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate sentence complexity (simplified)"""
        sentences = text.split('.')
        
        if not sentences:
            return 0.0
        
        complexities = []
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                # Complexity based on average word length and punctuation
                avg_word_len = sum(len(word) for word in words) / len(words)
                punctuation_count = sum(1 for char in sentence if char in ',;:!?')
                complexity = avg_word_len + (punctuation_count * 2)
                complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    
    def _create_readability_features(self, X: pd.DataFrame, text_columns: List[str],
                                   config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create readability features"""
        logger.info("📖 Creating readability features")
        
        readability_features = {}
        
        for col in text_columns:
            texts = X[col].fillna('').astype(str)
            
            # Readability metrics
            readability_features[f'{col}_automated_readability_index'] = texts.apply(
                self._calculate_ari
            )
            readability_features[f'{col}_coleman_liau_index'] = texts.apply(
                self._calculate_coleman_liau
            )
            readability_features[f'{col}_gunning_fog'] = texts.apply(
                self._calculate_gunning_fog
            )
        
        # Add features to dataframe
        for feature_name, feature_data in readability_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(readability_features.keys()),
            "readability_metrics": [
                "automated_readability_index", "coleman_liau_index", "gunning_fog"
            ]
        }
        
        return X, metadata
    
    def _calculate_ari(self, text: str) -> float:
        """Calculate Automated Readability Index (simplified)"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        chars = sum(len(word) for word in words)
        avg_chars_per_word = chars / len(words)
        avg_words_per_sentence = len(words) / len(sentences)
        
        ari = 4.71 * avg_chars_per_word + 0.5 * avg_words_per_sentence - 21.43
        return max(0, ari)
    
    def _calculate_coleman_liau(self, text: str) -> float:
        """Calculate Coleman-Liau Index (simplified)"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        chars = sum(len(word) for word in words)
        avg_chars_per_100_words = (chars / len(words)) * 100
        avg_sentences_per_100_words = (len(sentences) / len(words)) * 100
        
        cli = 0.0588 * avg_chars_per_100_words - 0.296 * avg_sentences_per_100_words - 15.8
        return max(0, cli)
    
    def _calculate_gunning_fog(self, text: str) -> float:
        """Calculate Gunning Fog Index (simplified)"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Count complex words (3+ syllables - simplified)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        avg_sentence_length = len(words) / len(sentences)
        percent_complex_words = (complex_words / len(words)) * 100
        
        fog = 0.4 * (avg_sentence_length + percent_complex_words)
        return max(0, fog)
    
    def _handle_text_missing(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in text features"""
        logger.info("🔧 Handling missing values in text features")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        missing_info = {}
        
        for col in numeric_cols:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    "missing_count": missing_count,
                    "missing_ratio": missing_count / len(X)
                }
                
                # Fill with 0 for text features
                X[col] = X[col].fillna(0)
        
        metadata = {
            "missing_info": missing_info,
            "total_missing_before": sum(X.isnull().sum()),
            "total_missing_after": sum(X.isnull().sum())
        }
        
        return X, metadata
    
    def _scale_text_features(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale text features"""
        logger.info("⚖️ Scaling text features")
        
        method = config.get("scaling_method", "standard")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            return X, {"method": "none"}
        
        # Fit and transform
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers["text_global"] = scaler
        
        metadata = {
            "method": method,
            "features_scaled": list(numeric_cols),
            "scaler_fitted": True
        }
        
        return X, metadata
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of text processing operations"""
        if not self.processing_history:
            return {"error": "No processing history available"}
        
        latest = self.processing_history[-1]
        
        return {
            "total_processing_sessions": len(self.processing_history),
            "latest_session": {
                "original_shape": latest["original_shape"],
                "final_shape": latest["final_shape"],
                "processing_time": latest["processing_time"],
                "steps_applied": latest["processing_steps"],
                "features_created": latest.get("features_created", 0)
            },
            "vocabulary_info": self.vocabulary,
            "all_steps_applied": list(set(
                step for session in self.processing_history 
                for step in session["processing_steps"]
            ))
        }
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new text data using fitted processors"""
        X_processed = X.copy()
        
        # Detect text columns
        text_columns = self._detect_text_columns(X_processed)
        
        if not text_columns:
            return X_processed
        
        # Apply text cleaning
        config = self._get_default_config()
        X_processed, _ = self._clean_text_data(X_processed, text_columns, config)
        
        # Apply scaling if fitted
        if "text_global" in self.scalers:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = self.scalers["text_global"].transform(X_processed[numeric_cols])
        
        return X_processed
    
    def reset(self):
        """Reset all fitted processors"""
        self.vectorizers.clear()
        self.scalers.clear()
        self.text_stats.clear()
        self.processing_history.clear()
        self.vocabulary.clear()
        logger.info("🔄 Text processor reset")
