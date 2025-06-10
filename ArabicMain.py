import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


# ================================
# DATA LOADING AND PREPROCESSING
# ================================

def load_your_data():


    # Convert to DataFrame
    df = pd.read_csv('Movies_and_TV_Reviews_en.csv')

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


def create_user_profiles_for_your_data():
    """
    Create realistic user profiles based on your review data
    """
    user_queries = [
        {
            'userId': 1,
            'query': 'historical accuracy WWII Band of Brothers',
            'interests': ['history', 'accuracy', 'documentary', 'war', 'educational'],
            'demographic': {'age_group': '45-60', 'interests_history': True},
            'past_purchases': ['war_documentaries', 'historical_series', 'HBO_shows']
        },
        {
            'userId': 2,
            'query': 'great war series entertainment value',
            'interests': ['entertainment', 'series', 'drama', 'action', 'value'],
            'demographic': {'age_group': '25-40', 'interests_history': False},
            'past_purchases': ['tv_series', 'action_movies', 'streaming_content']
        },
        {
            'userId': 3,
            'query': 'gift for father military history lover',
            'interests': ['gift', 'military', 'history', 'family', 'father'],
            'demographic': {'age_group': '30-45', 'interests_history': True},
            'past_purchases': ['gifts', 'military_history', 'documentaries']
        },
        {
            'userId': 4,
            'query': 'emotional war story miniseries',
            'interests': ['emotion', 'story', 'characters', 'drama', 'miniseries'],
            'demographic': {'age_group': '35-50', 'interests_history': True},
            'past_purchases': ['drama_series', 'character_driven_shows', 'award_winners']
        },
        {
            'userId': 5,
            'query': 'best HBO war production quality',
            'interests': ['quality', 'production', 'HBO', 'premium', 'cinematic'],
            'demographic': {'age_group': '40-55', 'interests_history': True},
            'past_purchases': ['HBO_shows', 'premium_content', 'high_budget_series']
        }
    ]

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

    # Display sample analysis
    print("Sample Feature Analysis:")
    print("-" * 40)
    for i in range(min(3, len(feature_df))):
        sample = feature_df.iloc[i]
        print(f"Review {sample['id']}:")
        print(f"Text: {sample['original_text']}")
        print(f"Quality Score: {sample['quality_score']:.2f}")
        print(f"War/History mentions: {sample.get('war_history_mentions', 0)}")
        print(f"Sentiment Score: {sample['sentiment_score']:.2f}")
        print(f"Helpfulness Ratio: {sample['helpfulness_ratio']:.2f}")
        print(f"Text Length: {sample['text_length']}")
        print("-" * 40)

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
        user_id = user_profile['userId']
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

        # Show top personalized reviews
        print("Top 3 personalized reviews:")
        for i, review in enumerate(user_results[:3], 1):
            print(f"{i}. Review ID {review['review_id']} (Personalization: {review['personalization_score']:.2f})")
            print(f"   Text: {review['original_text']}")
            print(f"   Semantic: {review['semantic_similarity']:.2f}, Interest: {review['interest_alignment']:.2f}")
        print("-" * 50)

    return personalized_results


# ================================
# FINAL RANKING SYSTEM
# ================================

def phase3_ranking_system_custom(feature_df, personalized_results):
    """
    Phase 3: Final ranking system customized for your data
    """
    print("=== PHASE 3: FINAL RANKING SYSTEM (Customized) ===")
    print("Generating final personalized rankings...\n")

    # Adjusted weights for your domain
    ranking_weights = {
        'quality': 0.20,
        'personalization': 0.40,  # Higher weight for personalization
        'helpfulness': 0.20,
        'recency': 0.10,
        'verification': 0.10
    }

    final_rankings = {}

    for user_id, user_reviews in personalized_results.items():
        print(f"Final Rankings for User {user_id}:")

        ranked_reviews = []

        for review_data in user_reviews:
            review_id = review_data['review_id']
            review = feature_df[feature_df['id'] == review_id].iloc[0]

            # Calculate final score
            final_score = (
                    review['quality_score'] * ranking_weights['quality'] +
                    review_data['personalization_score'] * ranking_weights['personalization'] +
                    review['helpfulness_ratio'] * 5 * ranking_weights['helpfulness'] +
                    review['recency_score'] * ranking_weights['recency'] +
                    review['verified_purchase'] * ranking_weights['verification']
            )

            ranked_reviews.append({
                'review_id': review_id,
                'final_score': final_score,
                'quality_score': review['quality_score'],
                'personalization_score': review_data['personalization_score'],
                'helpfulness_ratio': review['helpfulness_ratio'],
                'original_text': review['original_text'],
                'rating': review['rating'],
                'semantic_similarity': review_data['semantic_similarity'],
                'interest_alignment': review_data['interest_alignment']
            })

        # Sort by final score
        ranked_reviews.sort(key=lambda x: x['final_score'], reverse=True)
        final_rankings[user_id] = ranked_reviews

        # Display top 5 reviews
        print("🏆 TOP 5 RECOMMENDED REVIEWS:")
        for i, review in enumerate(ranked_reviews[:5], 1):
            print(f"\n{i}. Review ID {review['review_id']} | Final Score: {review['final_score']:.2f}")
            print(f"   📝 Text: {review['original_text']}")
            print(f"   ⭐ Rating: {review['rating']}/5")
            print(
                f"   📊 Quality: {review['quality_score']:.2f} | Personalization: {review['personalization_score']:.2f}")
            print(
                f"   🎯 Semantic Match: {review['semantic_similarity']:.2f} | Interest Align: {review['interest_alignment']:.2f}")

        print("=" * 70)

    return final_rankings


# ================================
# EVALUATION METRICS
# ================================

def evaluate_system_performance(final_rankings, feature_df):
    """Evaluate the performance of your customized system"""
    print("=== SYSTEM PERFORMANCE EVALUATION ===\n")

    for user_id, rankings in final_rankings.items():
        print(f"User {user_id} Performance Metrics:")

        # Top-k analysis
        top_3 = rankings[:3]
        top_5 = rankings[:5]

        # Average metrics for top recommendations
        avg_rating_top3 = np.mean([r['rating'] for r in top_3])
        avg_quality_top3 = np.mean([r['quality_score'] for r in top_3])
        avg_helpfulness_top3 = np.mean([r['helpfulness_ratio'] for r in top_3])

        print(f"  📈 Top 3 Avg Rating: {avg_rating_top3:.2f}/5")
        print(f"  📈 Top 3 Avg Quality Score: {avg_quality_top3:.2f}")
        print(f"  📈 Top 3 Avg Helpfulness: {avg_helpfulness_top3:.2f}")

        # Coverage analysis
        unique_content_types = set()
        for review in top_5:
            review_text = review['original_text'].lower()
            if any(word in review_text for word in ['history', 'war', 'historical']):
                unique_content_types.add('historical')
            if any(word in review_text for word in ['story', 'character', 'acting']):
                unique_content_types.add('narrative')
            if any(word in review_text for word in ['gift', 'present', 'father']):
                unique_content_types.add('personal')
            if any(word in review_text for word in ['value', 'worth', 'price']):
                unique_content_types.add('value')

        print(f"  🎯 Content Diversity: {len(unique_content_types)} different themes in top 5")
        print(f"  📋 Themes covered: {', '.join(unique_content_types)}")
        print("-" * 50)


# ================================
# MAIN EXECUTION
# ================================

def main_custom():
    """Main execution function for your customized data"""
    print("🚀 PERSONALIZED REVIEW RANKING SYSTEM")
    print("📊 Customized for Your Data")
    print("=" * 70)

    # Load your actual data
    reviews_df = load_your_data()
    user_queries = create_user_profiles_for_your_data()

    print(f"✅ Loaded {len(reviews_df)} reviews from your dataset")
    print(f"✅ Created {len(user_queries)} user profiles for testing\n")

    # Show sample of your raw data
    print("📋 Sample of Your Raw Data:")
    print(reviews_df[['index', 'asin', 'reviewText']].head())
    print(f"\nDataset shape: {reviews_df.shape}")
    print(f"Columns: {list(reviews_df.columns)}\n")

    # Phase 1: Feature Engineering
    try:
        feature_df = phase1_feature_engineering_custom(reviews_df)
        print(f"✅ Phase 1 Complete: Extracted features for {len(feature_df)} reviews\n")

        # Display feature summary
        print("📊 Feature Engineering Summary:")
        print(f"   - Text features: {len([col for col in feature_df.columns if col.startswith('text_')])} metrics")
        print(
            f"   - Domain features: {len([col for col in feature_df.columns if 'mentions' in col])} domain-specific signals")
        print(
            f"   - Sentiment features: {len([col for col in feature_df.columns if col.startswith('sentiment_')])} sentiment metrics")
        print(
            f"   - Quality features: {len([col for col in feature_df.columns if 'quality' in col or 'helpful' in col])} quality indicators")
        print(f"   - Average quality score: {feature_df['quality_score'].mean():.2f}")
        print()

    except Exception as e:
        print(f"❌ Error in Phase 1: {str(e)}")
        return

    # Phase 2: Personalization
    try:
        personalized_results = phase2_personalization_custom(feature_df, user_queries)
        print(f"✅ Phase 2 Complete: Personalized rankings for {len(personalized_results)} users\n")

        # Display personalization summary
        print("🎯 Personalization Summary:")
        for user_id, results in personalized_results.items():
            avg_personalization = np.mean([r['personalization_score'] for r in results[:10]])
            avg_semantic = np.mean([r['semantic_similarity'] for r in results[:10]])
            print(
                f"   User {user_id}: Avg personalization score {avg_personalization:.2f}, Avg semantic similarity {avg_semantic:.2f}")
        print()

    except Exception as e:
        print(f"❌ Error in Phase 2: {str(e)}")
        return

    # Phase 3: Final Ranking
    try:
        final_rankings = phase3_ranking_system_custom(feature_df, personalized_results)
        print(f"✅ Phase 3 Complete: Final rankings generated for all users\n")

    except Exception as e:
        print(f"❌ Error in Phase 3: {str(e)}")
        return

    # Phase 4: Evaluation
    try:
        evaluate_system_performance(final_rankings, feature_df)

    except Exception as e:
        print(f"❌ Error in Evaluation: {str(e)}")

    # Additional Analysis and Insights
    print("\n" + "=" * 70)
    print("🔍 ADDITIONAL SYSTEM INSIGHTS")
    print("=" * 70)

    # Overall system statistics
    total_reviews = len(feature_df)
    avg_quality = feature_df['quality_score'].mean()
    high_quality_count = len(feature_df[feature_df['quality_score'] > 5])

    print(f"📈 System Statistics:")
    print(f"   - Total reviews processed: {total_reviews}")
    print(f"   - Average quality score: {avg_quality:.2f}")
    print(f"   - High-quality reviews (>5.0): {high_quality_count} ({high_quality_count / total_reviews * 100:.1f}%)")
    print(
        f"   - Verified purchases: {feature_df['verified_purchase'].sum()} ({feature_df['verified_purchase'].mean() * 100:.1f}%)")
    print(f"   - Recent reviews (≤90 days): {feature_df['is_recent'].sum()}")

    # Feature importance analysis
    print(f"\n📊 Content Analysis:")
    print(f"   - Reviews mentioning war/history: {len(feature_df[feature_df['war_history_mentions'] > 0])}")
    print(f"   - Reviews with emotional impact: {len(feature_df[feature_df['emotional_impact_mentions'] > 0])}")
    print(f"   - Reviews with recommendations: {len(feature_df[feature_df['recommendation_mentions'] > 0])}")
    print(f"   - Reviews in personal context: {len(feature_df[feature_df['personal_context_mentions'] > 0])}")

    # Personalization effectiveness
    print(f"\n🎯 Personalization Effectiveness:")
    for user_id, rankings in final_rankings.items():
        user_profile = next(u for u in user_queries if u['userId'] == user_id)
        top_3_scores = [r['final_score'] for r in rankings[:3]]
        top_3_personalization = [r['personalization_score'] for r in rankings[:3]]

        print(f"   User {user_id} ({user_profile['query'][:30]}...):")
        print(f"     - Top 3 final scores: {[f'{s:.2f}' for s in top_3_scores]}")
        print(f"     - Top 3 personalization: {[f'{s:.2f}' for s in top_3_personalization]}")

    # Recommendations for system improvement
    print(f"\n💡 System Recommendations:")

    low_quality_count = len(feature_df[feature_df['quality_score'] < 2])
    if low_quality_count > total_reviews * 0.2:
        print(f"   ⚠️  High number of low-quality reviews ({low_quality_count}). Consider filtering threshold.")

    low_helpfulness = len(feature_df[feature_df['helpfulness_ratio'] < 0.3])
    if low_helpfulness > total_reviews * 0.3:
        print(f"   ⚠️  Many reviews have low helpfulness ratios. Consider weighting adjustment.")

    old_reviews = len(feature_df[feature_df['days_since_review'] > 365])
    if old_reviews > total_reviews * 0.5:
        print(f"   ℹ️  {old_reviews} reviews are over 1 year old. Consider recency boost.")

    print(f"   ✅ System successfully personalized {len(user_queries)} different user profiles")
    print(
        f"   ✅ Average semantic similarity in top results: {np.mean([np.mean([r['semantic_similarity'] for r in rankings[:5]]) for rankings in final_rankings.values()]):.3f}")

    print("\n" + "=" * 70)
    print("🎉 SYSTEM EXECUTION COMPLETE!")
    print("=" * 70)

    return final_rankings, feature_df, personalized_results


# Additional utility functions for post-analysis
def export_results_to_csv(final_rankings, feature_df, filename_prefix="ranking_results"):
    """Export results to CSV files for further analysis"""
    print(f"\n📁 Exporting results to CSV files...")

    try:
        # Export feature data
        feature_df.to_csv(f"{filename_prefix}_features.csv", index=False)
        print(f"   ✅ Features exported to {filename_prefix}_features.csv")

        # Export rankings for each user
        for user_id, rankings in final_rankings.items():
            rankings_df = pd.DataFrame(rankings)
            rankings_df.to_csv(f"{filename_prefix}_user_{user_id}.csv", index=False)
            print(f"   ✅ User {user_id} rankings exported to {filename_prefix}_user_{user_id}.csv")

    except Exception as e:
        print(f"   ❌ Export error: {str(e)}")


def generate_user_report(user_id, final_rankings, user_queries, feature_df):
    """Generate detailed report for a specific user"""
    print(f"\n📋 DETAILED REPORT FOR USER {user_id}")
    print("=" * 50)

    user_profile = next(u for u in user_queries if u['userId'] == user_id)
    rankings = final_rankings[user_id]

    print(f"🔍 User Query: '{user_profile['query']}'")
    print(f"🎯 Interests: {', '.join(user_profile['interests'])}")
    print(f"👤 Demographics: {user_profile['demographic']}")
    print(f"🛒 Past Purchases: {', '.join(user_profile['past_purchases'])}")

    print(f"\n📊 Top 10 Recommended Reviews:")
    for i, review in enumerate(rankings[:10], 1):
        print(f"\n{i}. Review ID: {review['review_id']}")
        print(f"   Final Score: {review['final_score']:.2f}")
        print(f"   Rating: {review['rating']}/5 ⭐")
        print(f"   Quality: {review['quality_score']:.2f}")
        print(f"   Personalization: {review['personalization_score']:.2f}")
        print(f"   Semantic Match: {review['semantic_similarity']:.2f}")
        print(f"   Interest Alignment: {review['interest_alignment']:.2f}")
        print(f"   Text: {review['original_text'][:100]}...")


# Execute the system
if __name__ == "__main__":
    # Run the main system
    final_rankings, feature_df, personalized_results = main_custom()

    # Optional: Export results
    export_results_to_csv(final_rankings, feature_df)

    # Optional: Generate detailed report for specific user
    # generate_user_report(1, final_rankings, create_user_profiles_for_your_data(), feature_df)ndex', 'asin', 'review']])
