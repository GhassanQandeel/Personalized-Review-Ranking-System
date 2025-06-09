import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import json
from flask import Flask, render_template, request, jsonify, session
import requests
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'g12g'

warnings.filterwarnings('ignore')
PRODUCTS = {
    '1': {'name': 'Game Of Thrones', 'price': 99.99, 'image': 'game-of-thrones.jpg'},
    '2': {'name': 'Breaking Bad', 'price': 699.99, 'image': 'breakingbad.jpeg'},
    '3': {'name': 'Lord of the rings ', 'price': 1299.99, 'image': 'lordoftherings.jpeg'},
    '4': {'name': 'The Godfather', 'price': 14.99, 'image': 'thegodfather.jpeg'},
    '5': {'name': 'The Hobbit', 'price': 9.99, 'image': 'thehobbit.jpeg'},
    '6': {'name': 'House of the dragon ', 'price': 24.99, 'image': 'houseofthedragon.jpeg'},

}


# ================================
# DATA LOADING AND PREPROCESSING
# ================================

def load_your_data(asin):
    # Convert to DataFrame
    df = pd.read_csv('Movies_and_TV_Reviews.csv')
    df.drop(columns=['reviewText_ar'], inplace=True)
    df = df[df['asin'] == asin]

    def estimate_rating(text):
        positive_indicators = ['love', 'great', 'best', 'excellent', 'amazing', 'perfect']
        negative_indicators = ['bad', 'terrible', 'poor', 'awful', 'disappointing']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_indicators if word in text_lower)
        neg_count = sum(1 for word in negative_indicators if word in text_lower)

        if pos_count > neg_count:
            return np.random.choice([4, 5], p=[0.3, 0.7])
        elif neg_count > pos_count:
            return np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        else:
            return np.random.choice([3, 4], p=[0.6, 0.4])

    # Estimate other missing fields
    df['rating'] = df['reviewText'].apply(estimate_rating)
    df['verified'] = np.random.choice([True, False], size=len(df), p=[0.8, 0.2])

    # Estimate helpfulness based on text length and content quality
    def estimate_helpfulness(text, rating):
        base_helpful = min(max(len(text) // 20, 1), 15)  # Based on text length
        if rating >= 4:
            base_helpful += np.random.randint(0, 5)
        helpful_votes = max(1, base_helpful + np.random.randint(-2, 3))
        total_votes = helpful_votes + np.random.randint(0, max(helpful_votes // 2, 1))
        return helpful_votes, total_votes

    helpfulness_data = df.apply(lambda row: estimate_helpfulness(row['reviewText'], row['rating']), axis=1)
    df['helpfulVotes'] = [h[0] for h in helpfulness_data]
    df['totalVotes'] = [h[1] for h in helpfulness_data]

    # Estimate timestamps (spread over last 2 years)
    start_date = datetime.now() - timedelta(days=730)
    df['timestamp'] = [
        (start_date + timedelta(days=np.random.randint(0, 730))).strftime('%Y-%m-%d')
        for _ in range(len(df))
    ]

    return df


def create_user_profiles_for_your_data(user_id):
    with open('user_profiles.json', 'r', encoding='utf-8') as f:
        user_profiles = json.load(f)
    user_queries = []
    for profile in user_profiles['user_profiles']:
        user_queries.append(profile)

    user_queries = [profile for profile in user_queries if profile['userId'] == user_id]
    return user_queries

# ================================
# ENHANCED FEATURE ENGINEERING FOR YOUR DATA
# ================================

def extract_text_features_enhanced(review_text):
    """Enhanced text feature extraction for your data"""
    if pd.isna(review_text) or review_text == '':
        return {f'text_{feature}': 0 for feature in ['length', 'word_count', 'sentence_count',
                                                     'avg_words_per_sentence', 'avg_word_length', 'exclamation_count',
                                                     'question_count', 'caps_ratio', 'punctuation_density']}

    words = review_text.split()
    sentences = re.split(r'[.!?]+', review_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Additional text quality indicators
    punctuation_count = len(re.findall(r'[.!?,:;]', review_text))

    return {
        'text_length': len(review_text),
        'text_word_count': len(words),
        'text_sentence_count': len(sentences),
        'text_avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'text_avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'text_exclamation_count': review_text.count('!'),
        'text_question_count': review_text.count('?'),
        'text_caps_ratio': sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1),
        'text_punctuation_density': punctuation_count / max(len(review_text), 1)
    }


def extract_domain_specific_features(review_text):
    """Extract features specific to your domain (war/history series)"""
    if pd.isna(review_text):
        review_text = ""

    text_lower = review_text.lower()

    # Domain-specific keywords
    domain_features = {
        'war_history': ['ww2', 'wwii', 'war', 'historical', 'history', 'military', 'battle', 'soldier'],
        'series_quality': ['series', 'episode', 'season', 'show', 'miniseries', 'hbo', 'production'],
        'emotional_impact': ['emotional', 'moving', 'touching', 'powerful', 'impact', 'feel', 'heart'],
        'story_narrative': ['story', 'narrative', 'character', 'plot', 'acting', 'performance'],
        'authenticity': ['accurate', 'authentic', 'real', 'true', 'realistic', 'detail', 'research'],
        'recommendation': ['recommend', 'must', 'should', 'watch', 'buy', 'get', 'worth'],
        'comparison': ['best', 'better', 'top', 'favorite', 'compared', 'like', 'similar'],
        'personal_context': ['father', 'dad', 'husband', 'family', 'gift', 'present', 'christmas'],
        'value_assessment': ['value', 'worth', 'price', 'money', 'cost', 'cheap', 'expensive']
    }

    features = {}
    for category, keywords in domain_features.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        features[f'{category}_mentions'] = count
        features[f'has_{category}'] = 1 if count > 0 else 0

    return features


def extract_sentiment_features_enhanced(review_text):
    """Enhanced sentiment analysis for your data"""
    if pd.isna(review_text):
        review_text = ""

    # Domain-specific positive and negative words
    positive_words = ['love', 'great', 'best', 'excellent', 'amazing', 'wonderful', 'fantastic',
                      'outstanding', 'perfect', 'incredible', 'brilliant', 'masterpiece',
                      'unbelievable', 'superb', 'phenomenal']

    negative_words = ['bad', 'terrible', 'awful', 'poor', 'disappointing', 'waste',
                      'worst', 'horrible', 'useless', 'boring', 'pathetic', 'annoying']

    # Intensity words
    intensity_words = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally']

    text_lower = review_text.lower()
    words = text_lower.split()

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    intensity_count = sum(1 for word in intensity_words if word in text_lower)

    # Calculate sentiment metrics
    sentiment_score = (pos_count - neg_count) / max(len(words), 1)
    sentiment_intensity = intensity_count / max(len(words), 1)

    return {
        'sentiment_positive_count': pos_count,
        'sentiment_negative_count': neg_count,
        'sentiment_intensity_count': intensity_count,
        'sentiment_score': sentiment_score,
        'sentiment_polarity': 1 if pos_count > neg_count else (-1 if neg_count > pos_count else 0),
        'sentiment_intensity': sentiment_intensity,
        'sentiment_strength': pos_count + neg_count
    }


def calculate_review_quality_score(features, verified, helpful_votes, total_votes):
    """Calculate comprehensive quality score"""

    # Base quality from text features
    text_quality = min(features['text_length'] / 50, 2)  # Cap at 2 points

    # Content richness
    content_richness = (
            features.get('war_history_mentions', 0) * 0.3 +
            features.get('story_narrative_mentions', 0) * 0.2 +
            features.get('authenticity_mentions', 0) * 0.3 +
            features.get('emotional_impact_mentions', 0) * 0.2
    )

    # Sentiment quality (positive reviews generally more helpful)
    sentiment_quality = max(features['sentiment_score'] * 2, 0)

    # Social proof
    helpfulness_score = (helpful_votes / max(total_votes, 1)) * 3
    verification_bonus = 1 if verified else 0

    # Combine all factors
    quality_score = (
            text_quality * 0.2 +
            content_richness * 0.3 +
            sentiment_quality * 0.2 +
            helpfulness_score * 0.2 +
            verification_bonus * 0.1
    )

    return quality_score


def phase1_feature_engineering_custom(df):
    """
    Phase 1: Feature engineering customized for your data
    """
    print("=== PHASE 1: FEATURE ENGINEERING (Customized for Your Data) ===")
    print(f"Processing {len(df)} reviews from your dataset...\n")

    features_list = []

    for idx, row in df.iterrows():
        # Start with basic info
        review_features = {
            'id': row['index'],  # Using 'index' from your data
            'asin': row['asin'],
            'original_text': row['reviewText'],
            'rating': row['rating']
        }

        # Extract all feature types
        text_features = extract_text_features_enhanced(row['reviewText'])
        domain_features = extract_domain_specific_features(row['reviewText'])
        sentiment_features = extract_sentiment_features_enhanced(row['reviewText'])

        # Quality indicators
        quality_features = {
            'verified_purchase': 1 if row['verified'] else 0,
            'helpfulness_ratio': row['helpfulVotes'] / max(row['totalVotes'], 1),
            'helpful_votes': row['helpfulVotes'],
            'total_votes': row['totalVotes']
        }

        # Recency features
        review_date = datetime.strptime(row['timestamp'], '%Y-%m-%d')
        days_diff = (datetime.now() - review_date).days
        recency_features = {
            'days_since_review': days_diff,
            'recency_score': np.exp(-days_diff / 365),
            'is_recent': 1 if days_diff <= 90 else 0
        }

        # Combine all features
        all_features = {**text_features, **domain_features, **sentiment_features,
                        **quality_features, **recency_features}
        review_features.update(all_features)

        # Calculate composite quality score
        quality_score = calculate_review_quality_score(
            all_features, row['verified'], row['helpfulVotes'], row['totalVotes']
        )
        review_features['quality_score'] = quality_score

        features_list.append(review_features)

    feature_df = pd.DataFrame(features_list)

    return feature_df


# ================================
# PERSONALIZATION FOR YOUR DATA
# ================================

def calculate_semantic_similarity_enhanced(review_text, user_query):
    """Enhanced semantic similarity calculation"""
    try:
        if pd.isna(review_text) or review_text == '' or user_query == '':
            return 0

        # Use TF-IDF with custom parameters for short texts
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2),  # Include bigrams
            min_df=1
        )

        texts = [review_text.lower(), user_query.lower()]
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback to simple word overlap
            review_words = set(review_text.lower().split())
            query_words = set(user_query.lower().split())
            overlap = len(review_words.intersection(query_words))
            return overlap / max(len(query_words), 1)
    except:
        return 0


def phase2_personalization_custom(feature_df, user_queries):
    """
    Phase 2: Personalization customized for your data
    """
    print("=== PHASE 2: PERSONALIZATION (Customized for Your Data) ===")
    print("Calculating personalized relevance scores...\n")

    personalized_results = {}

    for user_profile in user_queries:
        user_id = user_profile["userId"]
        print(f"Processing User {user_id}:")
        print(f"Query: '{user_profile['query']}'")
        print(f"Interests: {user_profile['interests']}")

        user_results = []

        for idx, review in feature_df.iterrows():
            # Semantic similarity
            semantic_score = calculate_semantic_similarity_enhanced(
                review['original_text'], user_profile['query']
            )

            # Interest alignment based on domain features
            interest_score = 0
            for interest in user_profile['interests']:
                interest_lower = interest.lower()

                # Map interests to feature columns
                if interest_lower in ['history', 'war', 'military']:
                    interest_score += review.get('war_history_mentions', 0) * 0.5
                elif interest_lower in ['entertainment', 'series', 'show']:
                    interest_score += review.get('series_quality_mentions', 0) * 0.5
                elif interest_lower in ['emotion', 'emotional', 'moving']:
                    interest_score += review.get('emotional_impact_mentions', 0) * 0.5
                elif interest_lower in ['story', 'narrative', 'character']:
                    interest_score += review.get('story_narrative_mentions', 0) * 0.5
                elif interest_lower in ['gift', 'father', 'family']:
                    interest_score += review.get('personal_context_mentions', 0) * 0.5
                elif interest_lower in ['quality', 'production', 'hbo']:
                    interest_score += review.get('series_quality_mentions', 0) * 0.5
                elif interest_lower in ['value', 'worth']:
                    interest_score += review.get('value_assessment_mentions', 0) * 0.5

            # Normalize interest score
            interest_score = interest_score / max(len(user_profile['interests']), 1)

            # Demographic matching
            demo_score = 0
            age_group = user_profile.get('demographic', {}).get('age_group', '')
            if age_group in ['45-60', '40-55', '35-50'] and review.get('war_history_mentions', 0) > 0:
                demo_score += 1

            # Purchase context matching
            context_score = 0
            past_purchases = user_profile.get('past_purchases', [])
            for purchase in past_purchases:
                if 'war' in purchase or 'military' in purchase or 'history' in purchase:
                    context_score += review.get('war_history_mentions', 0) * 0.3
                if 'hbo' in purchase.lower() or 'premium' in purchase:
                    context_score += review.get('series_quality_mentions', 0) * 0.3
                if 'gift' in purchase:
                    context_score += review.get('personal_context_mentions', 0) * 0.3

            context_score = context_score / max(len(past_purchases), 1)

            # Combined personalization score
            personalization_score = (
                    semantic_score * 3.0 +
                    interest_score * 2.5 +
                    demo_score * 1.0 +
                    context_score * 1.5
            )

            user_results.append({
                'review_id': review['id'],
                'semantic_similarity': semantic_score,
                'interest_alignment': interest_score,
                'demographic_match': demo_score,
                'context_similarity': context_score,
                'personalization_score': personalization_score,
                'original_text': review['original_text'],
                'quality_score': review['quality_score'],
                'rating': review['rating']
            })

        # Sort by personalization score
        user_results.sort(key=lambda x: x['personalization_score'], reverse=True)
        personalized_results[user_id] = user_results

    return personalized_results


# ================================
# ML-BASED RANKING SYSTEM (PHASE 3)
# ================================

def create_training_data(feature_df, personalized_results):
    """
    Create training data for the ranking model
    """
    print("🔧 Creating training data for ML ranking model...")

    training_data = []

    for user_id, user_reviews in personalized_results.items():
        for review_data in user_reviews:
            review_id = review_data['review_id']
            review = feature_df[feature_df['id'] == review_id].iloc[0]

            # Create synthetic relevance score based on multiple factors
            # This simulates user feedback/engagement data
            relevance_score = (
                    review['quality_score'] * 0.25 +
                    review_data['personalization_score'] * 0.35 +
                    review['helpfulness_ratio'] * 3 * 0.20 +
                    review['recency_score'] * 0.10 +
                    review['verified_purchase'] * 0.10 +
                    # Add some noise to make it more realistic
                    np.random.normal(0, 0.5)
            )

            # Clip to reasonable range
            relevance_score = np.clip(relevance_score, 0, 10)

            # Features for the model
            features = {
                'user_id': user_id,
                'quality_score': review['quality_score'],
                'personalization_score': review_data['personalization_score'],
                'semantic_similarity': review_data['semantic_similarity'],
                'interest_alignment': review_data['interest_alignment'],
                'helpfulness_ratio': review['helpfulness_ratio'],
                'helpful_votes': review['helpful_votes'],
                'total_votes': review['total_votes'],
                'recency_score': review['recency_score'],
                'verified_purchase': review['verified_purchase'],
                'rating': review['rating'],
                'text_length': review['text_length'],
                'text_word_count': review['text_word_count'],
                'sentiment_score': review['sentiment_score'],
                'sentiment_polarity': review['sentiment_polarity'],
                'war_history_mentions': review.get('war_history_mentions', 0),
                'emotional_impact_mentions': review.get('emotional_impact_mentions', 0),
                'story_narrative_mentions': review.get('story_narrative_mentions', 0),
                'authenticity_mentions': review.get('authenticity_mentions', 0),
                'recommendation_mentions': review.get('recommendation_mentions', 0),
                'personal_context_mentions': review.get('personal_context_mentions', 0),
                'relevance_score': relevance_score  # Target variable
            }

            training_data.append(features)

    training_df = pd.DataFrame(training_data)
    print(f"✅ Created training dataset with {len(training_df)} samples")
    print(f"   - Users: {training_df['user_id'].nunique()}")
    print(f"   - Features: {len(training_df.columns) - 2}")  # Exclude user_id and target
    print(f"   - Target range: {training_df['relevance_score'].min():.2f} - {training_df['relevance_score'].max():.2f}")

    return training_df


def train_ranking_models_ridge_only(training_df):
    """
    Train only Ridge Regression model for ranking
    """
    print("\n🤖 Training Ridge Regression ranking model...")

    # Prepare features and target
    feature_cols = [col for col in training_df.columns if col not in ['user_id', 'relevance_score']]
    X = training_df[feature_cols]
    y = training_df['relevance_score']

    # Handle any missing values
    X = X.fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features (Ridge requires scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Ridge Regression with different alpha values to test
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_alpha = 1.0
    best_cv_score = -np.inf

    print("🔍 Testing different Ridge alpha values...")

    # Find best alpha through cross-validation
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha, random_state=42)
        cv_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=2)
        cv_mean = cv_scores.mean()

        print(f"   Alpha {alpha}: CV R² = {cv_mean:.3f} (±{cv_scores.std():.3f})")

        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_alpha = alpha

    # Train final model with best alpha
    print(f"\n🏆 Best alpha: {best_alpha} (CV R²: {best_cv_score:.3f})")

    final_model = Ridge(alpha=best_alpha, random_state=42)
    final_model.fit(X_train_scaled, y_train)

    # Evaluate final model
    train_score = final_model.score(X_train_scaled, y_train)
    test_score = final_model.score(X_test_scaled, y_test)

    print(f"\n📊 Final Ridge Model Performance:")
    print(f"   📊 Train R²: {train_score:.3f}")
    print(f"   📊 Test R²: {test_score:.3f}")
    print(f"   📊 CV R²: {best_cv_score:.3f}")

    # Feature importance (coefficients for Ridge)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': final_model.coef_,
        'abs_coefficient': np.abs(final_model.coef_)
    }).sort_values('abs_coefficient', ascending=False)



    model_results = {
        'Ridge Regression': {
            'model': final_model,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': best_cv_score,
            'best_alpha': best_alpha,
            'feature_importance': feature_importance
        }
    }

    return final_model, scaler, feature_cols, model_results




def phase3_ml_ranking_system_ridge_only(feature_df, personalized_results):
    """
    Phase 3: ML-based ranking system using only Ridge Regression
    """
    print("=== PHASE 3: RIDGE REGRESSION RANKING SYSTEM ===")
    print("Using Ridge Regression to rank reviews...\n")

    # Create training data
    training_df = create_training_data(feature_df, personalized_results)

    # Train Ridge regression model only
    model, scaler, feature_cols, model_results = train_ranking_models_ridge_only(training_df)

    # Generate predictions for all user-review combinations
    print("\n🔮 Generating Ridge Regression rankings...")

    final_rankings = {}

    for user_id, user_reviews in personalized_results.items():
        print(f"\nRidge Regression Rankings for User {user_id}:")

        ranked_reviews = []

        # Prepare features for this user's reviews
        user_features = []
        for review_data in user_reviews:
            review_id = review_data['review_id']
            review = feature_df[feature_df['id'] == review_id].iloc[0]

            features = {
                'quality_score': review['quality_score'],
                'personalization_score': review_data['personalization_score'],
                'semantic_similarity': review_data['semantic_similarity'],
                'interest_alignment': review_data['interest_alignment'],
                'helpfulness_ratio': review['helpfulness_ratio'],
                'helpful_votes': review['helpful_votes'],
                'total_votes': review['total_votes'],
                'recency_score': review['recency_score'],
                'verified_purchase': review['verified_purchase'],
                'rating': review['rating'],
                'text_length': review['text_length'],
                'text_word_count': review['text_word_count'],
                'sentiment_score': review['sentiment_score'],
                'sentiment_polarity': review['sentiment_polarity'],
                'war_history_mentions': review.get('war_history_mentions', 0),
                'emotional_impact_mentions': review.get('emotional_impact_mentions', 0),
                'story_narrative_mentions': review.get('story_narrative_mentions', 0),
                'authenticity_mentions': review.get('authenticity_mentions', 0),
                'recommendation_mentions': review.get('recommendation_mentions', 0),
                'personal_context_mentions': review.get('personal_context_mentions', 0),
            }

            user_features.append(features)

        # Convert to DataFrame and ensure all features are present
        user_features_df = pd.DataFrame(user_features)
        for col in feature_cols:
            if col not in user_features_df.columns:
                user_features_df[col] = 0

        user_features_df = user_features_df[feature_cols].fillna(0)

        # Make predictions using scaled features (Ridge requires scaling)
        ml_scores = model.predict(scaler.transform(user_features_df))

        # Combine with review data
        for i, (review_data, ml_score) in enumerate(zip(user_reviews, ml_scores)):
            review_info = {
                'review_id': review_data['review_id'],
                'ridge_score': ml_score,  # Changed from ml_score to ridge_score for clarity
                'personalization_score': review_data['personalization_score'],
                'quality_score': feature_df[feature_df['id'] == review_data['review_id']]['quality_score'].iloc[0],
                'original_text': review_data['original_text'],
                'rating': review_data['rating'],
                'semantic_similarity': review_data['semantic_similarity'],
                'interest_alignment': review_data['interest_alignment']
            }
            ranked_reviews.append(review_info)

        # Sort by Ridge score
        ranked_reviews.sort(key=lambda x: x['ridge_score'], reverse=True)
        final_rankings[user_id] = ranked_reviews



    return final_rankings, model, scaler, feature_cols


def evaluate_ranking_systems(feature_df, personalized_results, ml_rankings):
        """
        Compare different ranking approaches
        """
        print("=== RANKING SYSTEM EVALUATION ===")
        print("Comparing different ranking approaches...\n")

        evaluation_results = {}

        for user_id in personalized_results.keys():
            print(f"📊 Evaluation for User {user_id}:")

            # Get rankings from different methods
            rule_based = personalized_results[user_id]  # Rule-based personalization
            ml_based = ml_rankings[user_id]  # ML-based ranking

            # Create baseline ranking (by rating and helpfulness)
            baseline_ranking = []
            for review_data in rule_based:
                review_id = review_data['review_id']
                review = feature_df[feature_df['id'] == review_id].iloc[0]
                baseline_score = review['rating'] * 0.5 + review['helpfulness_ratio'] * 2.5
                baseline_ranking.append({
                    'review_id': review_id,
                    'score': baseline_score,
                    'original_text': review_data['original_text']
                })

            baseline_ranking.sort(key=lambda x: x['score'], reverse=True)

            # Compare top 5 recommendations
            print("🔍 Top 5 Recommendations Comparison:")
            print("\nRule-based Personalization:")
            for i, review in enumerate(rule_based[:5], 1):
                print(f"{i}. ID {review['review_id']} (Score: {review['personalization_score']:.2f})")

            print("\nML-based Ranking:")
            for i, review in enumerate(ml_based[:5], 1):
                print(f"{i}. ID {review['review_id']} (Score: {review['ml_score']:.2f})")

            print("\nBaseline (Rating + Helpfulness):")
            for i, review in enumerate(baseline_ranking[:5], 1):
                print(f"{i}. ID {review['review_id']} (Score: {review['score']:.2f})")

            # Calculate ranking correlation
            rule_top5 = [r['review_id'] for r in rule_based[:5]]
            ml_top5 = [r['review_id'] for r in ml_based[:5]]
            baseline_top5 = [r['review_id'] for r in baseline_ranking[:5]]

            # Overlap analysis
            rule_ml_overlap = len(set(rule_top5) & set(ml_top5))
            rule_baseline_overlap = len(set(rule_top5) & set(baseline_top5))
            ml_baseline_overlap = len(set(ml_top5) & set(baseline_top5))

            evaluation_results[user_id] = {
                'rule_ml_overlap': rule_ml_overlap,
                'rule_baseline_overlap': rule_baseline_overlap,
                'ml_baseline_overlap': ml_baseline_overlap,
                'rule_based_avg_score': np.mean([r['personalization_score'] for r in rule_based[:5]]),
                'ml_based_avg_score': np.mean([r['ml_score'] for r in ml_based[:5]]),
                'baseline_avg_score': np.mean([r['score'] for r in baseline_ranking[:5]])
            }

        return evaluation_results

    # ================================
    # RECOMMENDATION GENERATION
    # ================================

def generate_personalized_recommendations_ridge(user_id, ridge_rankings, feature_df, top_k=5):
    """
    Generate final personalized recommendations using Ridge Regression scores
    """
    print(f"🎯 RIDGE REGRESSION RECOMMENDATIONS FOR USER {user_id}")
    print("=" * 60)

    user_rankings = ridge_rankings[user_id]
    recommendations = []

    for i, review_data in enumerate(user_rankings[:top_k], 1):
        review_id = review_data['review_id']
        review = feature_df[feature_df['id'] == review_id].iloc[0]

        # Generate explanation
        explanation_factors = []

        if review_data['semantic_similarity'] > 0.3:
            explanation_factors.append("highly relevant to your search")

        if review_data['interest_alignment'] > 1.0:
            explanation_factors.append("matches your interests")

        if review['quality_score'] > 3.0:
            explanation_factors.append("high-quality detailed review")

        if review['helpfulness_ratio'] > 0.7:
            explanation_factors.append("found helpful by other users")

        if review['verified_purchase'] == 1:
            explanation_factors.append("from verified purchase")

        if review['rating'] >= 4:
            explanation_factors.append("positive rating")

        if review.get('war_history_mentions', 0) > 0:
            explanation_factors.append("discusses historical accuracy")

        if review.get('emotional_impact_mentions', 0) > 0:
            explanation_factors.append("mentions emotional impact")

        explanation = f"Recommended because it's {', '.join(explanation_factors[:3])}"
        if len(explanation_factors) > 3:
            explanation += f" and {len(explanation_factors) - 3} other factors"

        recommendation = {
            'rank': i,
            'review_id': review_id,
            'ridge_score': review_data['ridge_score'],  # Changed from ml_score
            'rating': review['rating'],
            'text': review['original_text'],
            'explanation': explanation,
            'key_metrics': {
                'quality_score': review['quality_score'],
                'helpfulness_ratio': review['helpfulness_ratio'],
                'semantic_similarity': review_data['semantic_similarity'],
                'personalization_score': review_data['personalization_score']
            }
        }

        recommendations.append(recommendation)




    return recommendations

    # ================================
    # MAIN EXECUTION FUNCTION
    # ================================
def run_complete_recommendation_system_ridge_only(user_id, asin):
    """
    Run the complete recommendation system with Ridge Regression only
    """
    global recommendations
    print("🚀 STARTING RECOMMENDATION SYSTEM")


    # Load and preprocess data
    print("📊 Loading your review data...")
    df = load_your_data(asin)
    print(f"✅ Loaded {len(df)} reviews successfully\n")

    # Create user profiles
    user_queries = create_user_profiles_for_your_data(user_id)

    # Phase 1: Feature Engineering
    feature_df = phase1_feature_engineering_custom(df)
    print(f"✅ Phase 1 Complete: Extracted features for {len(feature_df)} reviews\n")

    # Phase 2: Personalization
    personalized_results = phase2_personalization_custom(feature_df, user_queries)
    print(f"✅ Phase 2 Complete: Generated personalized rankings\n")

    # Phase 3: Ridge Regression Ranking
    ridge_rankings, model, scaler, feature_cols = phase3_ml_ranking_system_ridge_only(
        feature_df, personalized_results
    )
    print(f"✅ Phase 3 Complete: Ridge Regression ranking system trained and applied\n")

    # Generate final recommendations
    print("🎯 GENERATING FINAL RIDGE REGRESSION RECOMMENDATIONS")
    print("=" * 70)

    for user_profile in user_queries:
        user_id_num = user_profile['userId']
        print(f"\n👤 User {user_id_num} Query: '{user_profile['query']}'")
        print(f"Interests: {user_profile['interests']}")

        recommendations = generate_personalized_recommendations_ridge(
            user_id_num, ridge_rankings, feature_df, top_k=3
        )

    return recommendations





df_ar=pd.read_csv('Movies_and_TV_Reviews.csv')

def get_asin_by_value(product_dict, target_value):
    for asin, value in product_dict.items():
        if value == target_value:
            return asin
    return None  # if not found



@app.route('/')
def home():
    """Home page showing available products"""
    return render_template('home.html', products=PRODUCTS)




@app.route('/product/<product_id>')
def product_page(product_id):
    """Product detail page with enhanced user handling"""
    product = PRODUCTS.get(product_id)
    print(product)
    products_asins = {
    '1': 'B00R8GUXPG',
    '2': 'B00PY4Q9OS',
    '3': 'B00Q0G2VXM',
    '4': 'B000WGWQG8',
    '5': 'B00YSG2ZPA',
    '6': 'B00006CXSS'
}
    asin = products_asins.get(product_id)

    if not product:
        return "Product not found", 404

    user_id = session.get('user_id')
    if not user_id:
        user_id = request.args.get('user_id', '2')  # Default to user 2
        session['user_id'] = user_id


    print(user_id)
    print(asin)
    recommendations = run_complete_recommendation_system_ridge_only(int(user_id), asin)

    reviews = []
    for review in recommendations:
        arabic_text = df_ar.loc[df_ar['index'] == review['review_id']]['reviewText_ar'].values[0]
        reviews.append({
            'rank': review['rank'],
            'text_en': review['text'],
            'text_ar': arabic_text,
            'review_id': review['review_id']
        })

    return render_template('product.html',
                           product=product,
                           product_id=product_id,
                           user_id=user_id,
                           reviews=reviews)

# Enhanced user setting route
@app.route('/set-user/<user_id>')
def set_user(user_id):
    """Set user ID in session with validation"""
    try:
        # Validate user_id is numeric
        user_id_int = int(user_id)
        session['user_id'] = str(user_id_int)
        return jsonify({
            'success': True,
            'message': f"User ID set to: {user_id_int}",
            'user_id': user_id_int
        })
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'User ID must be numeric'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)



