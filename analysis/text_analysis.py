"""
Text Analysis Module for Cosmetics Brand Analysis

This module handles comprehensive text analysis including:
- Text preprocessing and cleaning
- Keyword extraction using TF-IDF
- Sentiment analysis using VADER
- Word cloud generation
- Topic modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class TextAnalysis:
    """
    Class for comprehensive text analysis of customer feedback and expectations.
    """
    
    def __init__(self, dataframe, text_column):
        """
        Initialize Text Analysis class.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The dataset containing text data
        text_column : str
            Name of the column containing text data
        """
        self.df = dataframe.copy()
        self.text_column = text_column
        self.cleaned_texts = []
        self.keywords = {}
        self.sentiments = {}
        self.topics = {}
        
        if text_column not in self.df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        
        # Initialize text processing tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Set up stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Add custom stop words for cosmetics domain
        custom_stop_words = {
            'cosmetic', 'cosmetics', 'brand', 'product', 'products', 
            'use', 'using', 'like', 'good', 'great', 'nice', 'want', 'need'
        }
        self.stop_words.update(custom_stop_words)
    
    def clean_text(self, text, remove_numbers=True, remove_punctuation=True, 
                  to_lowercase=True, remove_stopwords=True, apply_stemming=False, 
                  apply_lemmatization=True):
        """
        Clean and preprocess text data.
        
        Parameters:
        -----------
        text : str
            Raw text to clean
        remove_numbers : bool
            Whether to remove numbers
        remove_punctuation : bool
            Whether to remove punctuation
        to_lowercase : bool
            Whether to convert to lowercase
        remove_stopwords : bool
            Whether to remove stop words
        apply_stemming : bool
            Whether to apply stemming
        apply_lemmatization : bool
            Whether to apply lemmatization
            
        Returns:
        --------
        str
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        if to_lowercase:
            text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stop words
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if apply_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove empty tokens and very short words
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_all_texts(self):
        """
        Preprocess all texts in the dataset.
        """
        print("🧹 TEXT PREPROCESSING")
        print("=" * 25)
        
        # Clean all texts
        print("📝 Cleaning text data...")
        self.df['Cleaned_Text'] = self.df[self.text_column].apply(self.clean_text)
        
        # Remove empty texts
        initial_count = len(self.df)
        self.df = self.df[self.df['Cleaned_Text'].str.len() > 0]
        final_count = len(self.df)
        
        print(f"✅ Text preprocessing completed")
        print(f"📊 Texts processed: {initial_count}")
        print(f"📊 Valid texts: {final_count}")
        print(f"📊 Removed: {initial_count - final_count} empty texts")
        
        # Store cleaned texts
        self.cleaned_texts = self.df['Cleaned_Text'].tolist()
        
        return self.cleaned_texts
    
    def extract_keywords_tfidf(self, max_features=100, ngram_range=(1, 2)):
        """
        Extract keywords using TF-IDF analysis.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features to extract
        ngram_range : tuple
            Range of n-grams to consider
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing keywords and their TF-IDF scores
        """
        print("\n🔍 KEYWORD EXTRACTION (TF-IDF)")
        print("=" * 35)
        
        if not self.cleaned_texts:
            print("❌ No cleaned texts available. Run preprocess_all_texts() first.")
            return None
        
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        try:
            # Fit and transform texts
            tfidf_matrix = tfidf.fit_transform(self.cleaned_texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword dataframe
            keywords_df = pd.DataFrame({
                'Keyword': feature_names,
                'TF_IDF_Score': mean_scores
            }).sort_values('TF_IDF_Score', ascending=False)
            
            # Store keywords
            self.keywords['tfidf'] = keywords_df
            
            print(f"📊 Top 20 Keywords (TF-IDF):")
            print(keywords_df.head(20).to_string(index=False))
            
            # Visualize top keywords
            plt.figure(figsize=(12, 8))
            top_20_keywords = keywords_df.head(20)
            plt.barh(range(len(top_20_keywords)), top_20_keywords['TF_IDF_Score'])
            plt.yticks(range(len(top_20_keywords)), top_20_keywords['Keyword'])
            plt.xlabel('TF-IDF Score')
            plt.title('Top 20 Keywords by TF-IDF Score')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('images/tfidf_keywords.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return keywords_df
            
        except Exception as e:
            print(f"❌ Error in TF-IDF analysis: {e}")
            return None
    
    def generate_word_cloud(self, max_words=100, save_path=None):
        """
        Generate and display word cloud from text data.
        
        Parameters:
        -----------
        max_words : int
            Maximum number of words in the cloud
        save_path : str, optional
            Path to save the word cloud image
        """
        print("\n☁️ WORD CLOUD GENERATION")
        print("=" * 28)
        
        if not self.cleaned_texts:
            print("❌ No cleaned texts available. Run preprocess_all_texts() first.")
            return
        
        # Combine all texts
        combined_text = ' '.join(self.cleaned_texts)
        
        if len(combined_text.strip()) == 0:
            print("❌ No text content available for word cloud")
            return
        
        try:
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=max_words,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate(combined_text)
            
            # Display word cloud
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Customer Expectations - Word Cloud', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Word cloud saved to {save_path}")
            else:
                plt.savefig('images/word_cloud.png', dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generating word cloud: {e}")
    
    def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis using VADER and TextBlob.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing sentiment scores
        """
        print("\n😊 SENTIMENT ANALYSIS")
        print("=" * 23)
        
        if self.text_column not in self.df.columns:
            print("❌ Text column not found")
            return None
        
        # VADER sentiment analysis
        print("📊 Analyzing sentiment with VADER...")
        vader_scores = []
        
        for text in self.df[self.text_column]:
            if pd.notna(text) and text != '':
                scores = self.sentiment_analyzer.polarity_scores(str(text))
                vader_scores.append(scores)
            else:
                vader_scores.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
        
        # Convert to DataFrame
        vader_df = pd.DataFrame(vader_scores)
        
        # Add to main dataframe
        self.df['Sentiment_Negative'] = vader_df['neg']
        self.df['Sentiment_Neutral'] = vader_df['neu']
        self.df['Sentiment_Positive'] = vader_df['pos']
        self.df['Sentiment_Compound'] = vader_df['compound']
        
        # Categorize sentiment
        def categorize_sentiment(compound_score):
            if compound_score >= 0.05:
                return 'Positive'
            elif compound_score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.df['Sentiment_Category'] = self.df['Sentiment_Compound'].apply(categorize_sentiment)
        
        # Store sentiment results
        self.sentiments = {
            'vader_scores': vader_df,
            'sentiment_distribution': self.df['Sentiment_Category'].value_counts(),
            'mean_scores': {
                'negative': vader_df['neg'].mean(),
                'neutral': vader_df['neu'].mean(),
                'positive': vader_df['pos'].mean(),
                'compound': vader_df['compound'].mean()
            }
        }
        
        # Display results
        print(f"📊 Sentiment Distribution:")
        sentiment_counts = self.df['Sentiment_Category'].value_counts()
        sentiment_pct = (sentiment_counts / len(self.df)) * 100
        
        for sentiment, count in sentiment_counts.items():
            pct = sentiment_pct[sentiment]
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        print(f"\n📊 Average Sentiment Scores:")
        for score_type, avg_score in self.sentiments['mean_scores'].items():
            print(f"  {score_type.title()}: {avg_score:.3f}")
        
        # Visualize sentiment analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution pie chart
        sentiment_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
        axes[0,0].set_title('Sentiment Distribution')
        axes[0,0].set_ylabel('')
        
        # 2. Compound score distribution
        axes[0,1].hist(self.df['Sentiment_Compound'], bins=20, alpha=0.7, color='skyblue')
        axes[0,1].set_title('Compound Sentiment Score Distribution')
        axes[0,1].set_xlabel('Compound Score')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Sentiment components
        sentiment_components = self.df[['Sentiment_Negative', 'Sentiment_Neutral', 'Sentiment_Positive']].mean()
        sentiment_components.plot(kind='bar', ax=axes[1,0], color=['red', 'gray', 'green'])
        axes[1,0].set_title('Average Sentiment Components')
        axes[1,0].set_ylabel('Average Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Sentiment by text length
        self.df['Text_Length'] = self.df[self.text_column].str.len()
        sns.boxplot(data=self.df, x='Sentiment_Category', y='Text_Length', ax=axes[1,1])
        axes[1,1].set_title('Text Length by Sentiment')
        
        plt.tight_layout()
        plt.savefig('images/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df[['Sentiment_Negative', 'Sentiment_Neutral', 
                       'Sentiment_Positive', 'Sentiment_Compound', 'Sentiment_Category']]
    
    def topic_modeling(self, n_topics=5, max_features=100):
        """
        Perform topic modeling using Latent Dirichlet Allocation (LDA).
        
        Parameters:
        -----------
        n_topics : int
            Number of topics to extract
        max_features : int
            Maximum number of features for topic modeling
            
        Returns:
        --------
        dict
            Dictionary containing topic modeling results
        """
        print(f"\n📚 TOPIC MODELING (LDA - {n_topics} topics)")
        print("=" * 40)
        
        if not self.cleaned_texts:
            print("❌ No cleaned texts available. Run preprocess_all_texts() first.")
            return None
        
        try:
            # Vectorize texts for topic modeling
            vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            
            doc_term_matrix = vectorizer.fit_transform(self.cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract topics
            topics = {}
            n_top_words = 10
            
            print(f"📊 Discovered Topics:")
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics[f'Topic_{topic_idx + 1}'] = {
                    'words': top_words,
                    'weights': top_weights
                }
                
                print(f"\n  Topic {topic_idx + 1}:")
                for word, weight in zip(top_words[:5], top_weights[:5]):
                    print(f"    {word}: {weight:.3f}")
            
            # Visualize topics
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (topic_name, topic_data) in enumerate(topics.items()):
                if i < len(axes):
                    words = topic_data['words'][:8]
                    weights = topic_data['weights'][:8]
                    
                    axes[i].barh(range(len(words)), weights)
                    axes[i].set_yticks(range(len(words)))
                    axes[i].set_yticklabels(words)
                    axes[i].set_title(topic_name)
                    axes[i].invert_yaxis()
            
            # Hide unused subplot
            if len(topics) < len(axes):
                axes[-1].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('images/topic_modeling.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Store results
            self.topics = {
                'topics': topics,
                'model': lda,
                'vectorizer': vectorizer,
                'n_topics': n_topics
            }
            
            return topics
            
        except Exception as e:
            print(f"❌ Error in topic modeling: {e}")
            return None
    
    def analyze_text_length_patterns(self):
        """
        Analyze patterns in text length and their relationship with sentiment.
        """
        print("\n📏 TEXT LENGTH ANALYSIS")
        print("=" * 28)
        
        # Calculate text statistics
        self.df['Original_Text_Length'] = self.df[self.text_column].str.len()
        self.df['Word_Count'] = self.df[self.text_column].str.split().str.len()
        
        # Basic statistics
        print(f"📊 Text Length Statistics:")
        length_stats = self.df['Original_Text_Length'].describe()
        print(length_stats)
        
        print(f"\n📊 Word Count Statistics:")
        word_stats = self.df['Word_Count'].describe()
        print(word_stats)
        
        # Visualize text length distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Text length histogram
        axes[0].hist(self.df['Original_Text_Length'], bins=20, alpha=0.7, color='skyblue')
        axes[0].set_title('Text Length Distribution')
        axes[0].set_xlabel('Character Count')
        axes[0].set_ylabel('Frequency')
        
        # Word count histogram
        axes[1].hist(self.df['Word_Count'], bins=20, alpha=0.7, color='lightgreen')
        axes[1].set_title('Word Count Distribution')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        
        # Length vs sentiment (if sentiment analysis has been performed)
        if 'Sentiment_Compound' in self.df.columns:
            axes[2].scatter(self.df['Original_Text_Length'], self.df['Sentiment_Compound'], alpha=0.6)
            axes[2].set_xlabel('Text Length')
            axes[2].set_ylabel('Sentiment Score')
            axes[2].set_title('Text Length vs Sentiment')
            
            # Add correlation
            corr = self.df['Original_Text_Length'].corr(self.df['Sentiment_Compound'])
            axes[2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=axes[2].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[2].text(0.5, 0.5, 'Sentiment analysis not performed', 
                        ha='center', va='center', fontsize=12)
            axes[2].set_title('Text Length vs Sentiment')
        
        plt.tight_layout()
        plt.savefig('images/text_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_text_insights(self):
        """
        Generate comprehensive insights from text analysis.
        
        Returns:
        --------
        dict
            Dictionary containing key text analysis insights
        """
        insights = {}
        
        # Text statistics
        if 'Original_Text_Length' in self.df.columns:
            insights['avg_text_length'] = self.df['Original_Text_Length'].mean()
            insights['avg_word_count'] = self.df['Word_Count'].mean()
        
        # Sentiment insights
        if self.sentiments:
            insights['sentiment_distribution'] = self.sentiments['sentiment_distribution'].to_dict()
            insights['avg_sentiment'] = self.sentiments['mean_scores']['compound']
        
        # Keyword insights
        if self.keywords.get('tfidf') is not None:
            top_keywords = self.keywords['tfidf'].head(10)['Keyword'].tolist()
            insights['top_keywords'] = top_keywords
        
        # Topic insights
        if self.topics:
            insights['n_topics'] = self.topics['n_topics']
            insights['topic_words'] = {
                name: data['words'][:3] for name, data in self.topics['topics'].items()
            }
        
        print("\n💡 KEY TEXT ANALYSIS INSIGHTS:")
        print("=" * 35)
        for key, value in insights.items():
            print(f"{key}: {value}")
        
        return insights
    
    def export_text_analysis_report(self, filename="text_analysis_report.txt"):
        """
        Export text analysis results to a text file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        with open(filename, 'w') as f:
            f.write("COSMETICS BRAND ANALYSIS - TEXT ANALYSIS REPORT\n")
            f.write("=" * 58 + "\n\n")
            
            # Text statistics
            if 'Original_Text_Length' in self.df.columns:
                f.write("TEXT STATISTICS:\n")
                f.write("-" * 15 + "\n")
                f.write(f"Average text length: {self.df['Original_Text_Length'].mean():.1f} characters\n")
                f.write(f"Average word count: {self.df['Word_Count'].mean():.1f} words\n")
                f.write(f"Total responses: {len(self.df)}\n\n")
            
            # Sentiment analysis
            if self.sentiments:
                f.write("SENTIMENT ANALYSIS:\n")
                f.write("-" * 18 + "\n")
                for sentiment, count in self.sentiments['sentiment_distribution'].items():
                    pct = (count / len(self.df)) * 100
                    f.write(f"{sentiment}: {count} ({pct:.1f}%)\n")
                f.write(f"\nAverage compound sentiment: {self.sentiments['mean_scores']['compound']:.3f}\n\n")
            
            # Top keywords
            if self.keywords.get('tfidf') is not None:
                f.write("TOP KEYWORDS (TF-IDF):\n")
                f.write("-" * 20 + "\n")
                top_keywords = self.keywords['tfidf'].head(15)
                for _, row in top_keywords.iterrows():
                    f.write(f"{row['Keyword']}: {row['TF_IDF_Score']:.3f}\n")
                f.write("\n")
            
            # Topics
            if self.topics:
                f.write(f"TOPIC MODELING ({self.topics['n_topics']} topics):\n")
                f.write("-" * 25 + "\n")
                for topic_name, topic_data in self.topics['topics'].items():
                    f.write(f"{topic_name}: {', '.join(topic_data['words'][:5])}\n")
        
        print(f"✅ Text analysis report exported to {filename}")


def main():
    """
    Main function to demonstrate text analysis workflow.
    """
    print("🧴 COSMETICS BRAND ANALYSIS - TEXT ANALYSIS")
    print("=" * 50)
    
    print("📁 This module requires data to be loaded first.")
    print("Please run data_preparation.py first or import the DataPreparation class.")
    
    # Example usage:
    # from data_preparation import DataPreparation
    # data_prep = DataPreparation("data/your_file.csv")
    # df = data_prep.load_data()
    # 
    # text_analysis = TextAnalysis(df, 'Open_Feedback')
    # text_analysis.preprocess_all_texts()
    # text_analysis.extract_keywords_tfidf()
    # text_analysis.generate_word_cloud()
    # text_analysis.perform_sentiment_analysis()
    # text_analysis.topic_modeling()
    # text_analysis.analyze_text_length_patterns()
    # text_analysis.generate_text_insights()
    # text_analysis.export_text_analysis_report()


if __name__ == "__main__":
    main()