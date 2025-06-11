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
    """
    Load Arabic review data and create missing fields.
    Reads from 'reviewText_ar' column.
    """
    # Make sure the file name and path are correct
    # The relevant column is now 'reviewText_ar'
    df = pd.read_csv('Movies_and_TV_Reviews_ar.csv')

    def estimate_rating(text):
        # Arabic positive and negative indicators
        positive_indicators = ['أحببت', 'رائع', 'أفضل', 'ممتاز', 'مذهل', 'مثالي', 'تحفة']
        negative_indicators = ['سيء', 'فظيع', 'ضعيف', 'مروع', 'مخيب للآمال', 'ممل']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_indicators if word in text_lower)
        neg_count = sum(1 for word in negative_indicators if word in text_lower)

        if pos_count > neg_count:
            return np.random.choice([4, 5], p=[0.3, 0.7])
        elif neg_count > pos_count:
            return np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        else:
            return np.random.choice([3, 4], p=[0.6, 0.4])

    # Estimate missing fields using the 'reviewText_ar' column
    df['rating'] = df['reviewText_ar'].apply(estimate_rating)
    df['verified'] = np.random.choice([True, False], size=len(df), p=[0.8, 0.2])

    def estimate_helpfulness(text, rating):
        base_helpful = min(max(len(text) // 20, 1), 15)
        if rating >= 4:
            base_helpful += np.random.randint(0, 5)
        helpful_votes = max(1, base_helpful + np.random.randint(-2, 3))
        total_votes = helpful_votes + np.random.randint(0, max(helpful_votes // 2, 1))
        return helpful_votes, total_votes

    # Apply helpfulness estimation using 'reviewText_ar'
    helpfulness_data = df.apply(lambda row: estimate_helpfulness(row['reviewText_ar'], row['rating']), axis=1)
    df['helpfulVotes'] = [h[0] for h in helpfulness_data]
    df['totalVotes'] = [h[1] for h in helpfulness_data]

    start_date = datetime.now() - timedelta(days=730)
    df['timestamp'] = [
        (start_date + timedelta(days=np.random.randint(0, 730))).strftime('%Y-%m-%d')
        for _ in range(len(df))
    ]

    return df


def create_user_profiles_for_your_data():
    """
    Create realistic user profiles with Arabic queries and interests.
    """
    user_queries = [
        {
            'userId': 1,
            'query': 'الدقة التاريخية في مسلسل Band of Brothers عن الحرب العالمية الثانية',
            'interests': ['تاريخ', 'دقة', 'وثائقي', 'حرب', 'تعليمي'],
            'demographic': {'age_group': '45-60', 'interests_history': True},
            'past_purchases': ['وثائقيات_حربية', 'مسلسلات_تاريخية', 'إنتاجات_HBO']
        },
        {
            'userId': 2,
            'query': 'مسلسلات حربية عظيمة من حيث القيمة الترفيهية',
            'interests': ['ترفيه', 'مسلسلات', 'دراما', 'أكشن', 'قيمة'],
            'demographic': {'age_group': '25-40', 'interests_history': False},
            'past_purchases': ['مسلسلات_تلفزيونية', 'أفلام_أكشن', 'محتوى_رقمي']
        },
        {
            'userId': 3,
            'query': 'هدية لوالدي محب للتاريخ العسكري',
            'interests': ['هدية', 'عسكري', 'تاريخ', 'عائلة', 'والد'],
            'demographic': {'age_group': '30-45', 'interests_history': True},
            'past_purchases': ['هدايا', 'تاريخ_عسكري', 'وثائقيات']
        },
        {
            'userId': 4,
            'query': 'مسلسل قصير عن قصة حرب عاطفية',
            'interests': ['عاطفة', 'قصة', 'شخصيات', 'دراما', 'مسلسل_قصير'],
            'demographic': {'age_group': '35-50', 'interests_history': True},
            'past_purchases': ['مسلسلات_درامية', 'أعمال_تركز_على_الشخصيات', 'أعمال_حائزة_على_جوائز']
        },
        {
            'userId': 5,
            'query': 'أفضل إنتاجات HBO الحربية من حيث الجودة الإنتاجية',
            'interests': ['جودة', 'إنتاج', 'HBO', 'فاخر', 'سينمائي'],
            'demographic': {'age_group': '40-55', 'interests_history': True},
            'past_purchases': ['إنتاجات_HBO', 'محتوى_مميز', 'مسلسلات_بميزانية_ضخمة']
        }
    ]
    return user_queries


# ================================
# ENHANCED FEATURE ENGINEERING FOR YOUR DATA
# ================================

def extract_text_features_enhanced(review_text):
    """Enhanced text feature extraction for Arabic data."""
    if pd.isna(review_text) or review_text == '':
        return {f'text_{feature}': 0 for feature in ['length', 'word_count', 'sentence_count',
                                                     'avg_words_per_sentence', 'avg_word_length', 'exclamation_count',
                                                     'question_count', 'caps_ratio', 'punctuation_density']}

    words = review_text.split()
    sentences = re.split(r'[.!?]+', review_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    punctuation_count = len(re.findall(r'[.!؟،:؛]', review_text))

    return {
        'text_length': len(review_text),
        'text_word_count': len(words),
        'text_sentence_count': len(sentences),
        'text_avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'text_avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'text_exclamation_count': review_text.count('!'),
        'text_question_count': review_text.count('؟'),
        'text_caps_ratio': 0,  # Not applicable for Arabic
        'text_punctuation_density': punctuation_count / max(len(review_text), 1)
    }


def extract_domain_specific_features(review_text):
    """Extract domain-specific features using Arabic keywords."""
    if pd.isna(review_text):
        review_text = ""

    text_lower = review_text.lower()
    # Arabic domain keywords
    domain_features = {
        'war_history': ['حرب', 'تاريخي', 'عسكري', 'معركة', 'جندي', 'الحرب العالمية'],
        'series_quality': ['مسلسل', 'حلقة', 'موسم', 'عرض', 'إنتاج', 'جودة'],
        'emotional_impact': ['عاطفي', 'مؤثر', 'قوي', 'مشاعر', 'قلب'],
        'story_narrative': ['قصة', 'سرد', 'شخصية', 'حبكة', 'تمثيل', 'أداء'],
        'authenticity': ['دقيق', 'واقعي', 'حقيقي', 'تفاصيل', 'بحث'],
        'recommendation': ['أنصح', 'يجب', 'شاهد', 'اشتر', 'يستحق'],
        'comparison': ['أفضل', 'أحسن', 'أروع', 'مثل', 'شبيه بـ'],
        'personal_context': ['أبي', 'والدي', 'زوجي', 'عائلة', 'هدية', 'عيد'],
        'value_assessment': ['قيمة', 'يستحق', 'سعر', 'مال', 'رخيص', 'غالي']
    }

    features = {}
    for category, keywords in domain_features.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        features[f'{category}_mentions'] = count
        features[f'has_{category}'] = 1 if count > 0 else 0
    return features


def extract_sentiment_features_enhanced(review_text):
    """Enhanced sentiment analysis using Arabic keywords."""
    if pd.isna(review_text):
        review_text = ""

    # Arabic sentiment keywords
    positive_words = ['حب', 'رائع', 'أفضل', 'ممتاز', 'مذهل', 'عظيم', 'خيالي',
                      'مدهش', 'مثالي', 'لا يصدق', 'عبقري', 'تحفة', 'أسطوري']
    negative_words = ['سيء', 'فظيع', 'مروع', 'ضعيف', 'مخيب', 'مضيعة',
                      'أسوأ', 'رهيب', 'تافه', 'ممل', 'مزعج']
    intensity_words = ['جداً', 'للغاية', 'بشكل لا يصدق', 'تماماً', 'بالكامل', 'كلياً']

    text_lower = review_text.lower()
    words = text_lower.split()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    intensity_count = sum(1 for word in intensity_words if word in text_lower)
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
    """Calculate comprehensive quality score."""
    text_quality = min(features['text_length'] / 50, 2)
    content_richness = (
            features.get('war_history_mentions', 0) * 0.3 +
            features.get('story_narrative_mentions', 0) * 0.2 +
            features.get('authenticity_mentions', 0) * 0.3 +
            features.get('emotional_impact_mentions', 0) * 0.2
    )
    sentiment_quality = max(features['sentiment_score'] * 2, 0)
    helpfulness_score = (helpful_votes / max(total_votes, 1)) * 3
    verification_bonus = 1 if verified else 0

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
    Phase 1: Feature engineering customized for Arabic data from 'reviewText_ar'.
    """
    print("=== PHASE 1: FEATURE ENGINEERING (Customized for Arabic Data) ===")
    print(f"Processing {len(df)} reviews from your dataset...\n")

    features_list = []
    for idx, row in df.iterrows():
        review_text = row['reviewText_ar'] # Use the correct column
        review_features = {
            'id': row['index'],
            'asin': row['asin'],
            'original_text': review_text,
            'rating': row['rating']
        }
        text_features = extract_text_features_enhanced(review_text)
        domain_features = extract_domain_specific_features(review_text)
        sentiment_features = extract_sentiment_features_enhanced(review_text)
        quality_features = {
            'verified_purchase': 1 if row['verified'] else 0,
            'helpfulness_ratio': row['helpfulVotes'] / max(row['totalVotes'], 1),
            'helpful_votes': row['helpfulVotes'],
            'total_votes': row['totalVotes']
        }
        review_date = datetime.strptime(row['timestamp'], '%Y-%m-%d')
        days_diff = (datetime.now() - review_date).days
        recency_features = {
            'days_since_review': days_diff,
            'recency_score': np.exp(-days_diff / 365),
            'is_recent': 1 if days_diff <= 90 else 0
        }
        all_features = {**text_features, **domain_features, **sentiment_features,
                        **quality_features, **recency_features}
        review_features.update(all_features)
        quality_score = calculate_review_quality_score(
            all_features, row['verified'], row['helpfulVotes'], row['totalVotes']
        )
        review_features['quality_score'] = quality_score
        features_list.append(review_features)

    feature_df = pd.DataFrame(features_list)
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
    """Enhanced semantic similarity for Arabic text."""
    try:
        if pd.isna(review_text) or review_text == '' or user_query == '':
            return 0
        # TF-IDF adapted for Arabic (no English stop words)
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1
        )
        texts = [review_text.lower(), user_query.lower()]
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            review_words = set(review_text.lower().split())
            query_words = set(user_query.lower().split())
            overlap = len(review_words.intersection(query_words))
            return overlap / max(len(query_words), 1)
    except:
        return 0


def phase2_personalization_custom(feature_df, user_queries):
    """
    Phase 2: Personalization using Arabic profiles and content.
    """
    print("=== PHASE 2: PERSONALIZATION (Customized for Arabic Data) ===")
    print("Calculating personalized relevance scores...\n")

    personalized_results = {}
    for user_profile in user_queries:
        user_id = user_profile['userId']
        print(f"Processing User {user_id}:")
        print(f"Query: '{user_profile['query']}'")
        print(f"Interests: {user_profile['interests']}")

        user_results = []
        for idx, review in feature_df.iterrows():
            semantic_score = calculate_semantic_similarity_enhanced(
                review['original_text'], user_profile['query']
            )
            interest_score = 0
            for interest in user_profile['interests']:
                interest_lower = interest.lower()
                # Logic matching Arabic interests to Arabic features
                if interest_lower in ['تاريخ', 'حرب', 'عسكري']:
                    interest_score += review.get('war_history_mentions', 0) * 0.5
                elif interest_lower in ['ترفيه', 'مسلسلات', 'عرض']:
                    interest_score += review.get('series_quality_mentions', 0) * 0.5
                elif interest_lower in ['عاطفة', 'عاطفي', 'مؤثر']:
                    interest_score += review.get('emotional_impact_mentions', 0) * 0.5
                # ... and so on for other Arabic keywords

            interest_score = interest_score / max(len(user_profile['interests']), 1)
            demo_score = 0
            age_group = user_profile.get('demographic', {}).get('age_group', '')
            if age_group in ['45-60', '40-55', '35-50'] and review.get('war_history_mentions', 0) > 0:
                demo_score += 1

            # ... (context and personalization score calculation remains the same)

            personalization_score = (
                    semantic_score * 3.0 +
                    interest_score * 2.5 +
                    demo_score * 1.0
            )
            user_results.append({
                'review_id': review['id'],
                'semantic_similarity': semantic_score,
                'interest_alignment': interest_score,
                'personalization_score': personalization_score,
                'original_text': review['original_text'],
                'quality_score': review['quality_score'],
                'rating': review['rating']
            })

        user_results.sort(key=lambda x: x['personalization_score'], reverse=True)
        personalized_results[user_id] = user_results

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
    Phase 3: Final ranking system with English labels.
    """
    print("=== PHASE 3: FINAL RANKING SYSTEM (Customized) ===")
    print("Generating final personalized rankings...\n")

    ranking_weights = {
        'quality': 0.20,
        'personalization': 0.40,
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

        ranked_reviews.sort(key=lambda x: x['final_score'], reverse=True)
        final_rankings[user_id] = ranked_reviews

        print("🏆 TOP 5 RECOMMENDED REVIEWS:")
        for i, review in enumerate(ranked_reviews[:5], 1):
            print(f"\n{i}. Review ID {review['review_id']} | Final Score: {review['final_score']:.2f}")
            print(f"   📝 Text: {review['original_text']}")
            print(f"   ⭐ Rating: {review['rating']}/5")
            print(f"   📊 Quality: {review['quality_score']:.2f} | Personalization: {review['personalization_score']:.2f}")
            print(f"   🎯 Semantic Match: {review['semantic_similarity']:.2f} | Interest Align: {review['interest_alignment']:.2f}")
        print("=" * 70)

    return final_rankings


# ================================
# EVALUATION METRICS
# ================================

def evaluate_system_performance(final_rankings, feature_df):
    """Evaluate the performance of the system with English labels."""
    print("=== SYSTEM PERFORMANCE EVALUATION ===\n")

    for user_id, rankings in final_rankings.items():
        print(f"User {user_id} Performance Metrics:")
        top_3 = rankings[:3]
        top_5 = rankings[:5]
        avg_rating_top3 = np.mean([r['rating'] for r in top_3])
        avg_quality_top3 = np.mean([r['quality_score'] for r in top_3])
        avg_helpfulness_top3 = np.mean([r['helpfulness_ratio'] for r in top_3])

        print(f"   📈 Top 3 Avg Rating: {avg_rating_top3:.2f}/5")
        print(f"   📈 Top 3 Avg Quality Score: {avg_quality_top3:.2f}")
        print(f"   📈 Top 3 Avg Helpfulness: {avg_helpfulness_top3:.2f}")

        unique_content_types = set()
        for review in top_5:
            review_text = review['original_text'].lower()
            # Use Arabic keywords for analysis
            if any(word in review_text for word in ['تاريخ', 'حرب']):
                unique_content_types.add('historical')
            if any(word in review_text for word in ['قصة', 'شخصية', 'تمثيل']):
                unique_content_types.add('narrative')
            if any(word in review_text for word in ['هدية', 'أب', 'والدي']):
                unique_content_types.add('personal')
            if any(word in review_text for word in ['قيمة', 'يستحق', 'سعر']):
                unique_content_types.add('value')

        print(f"   🎯 Content Diversity: {len(unique_content_types)} different themes in top 5")
        print(f"   📋 Themes covered: {', '.join(unique_content_types)}")
        print("-" * 50)

# ================================
# UTILITY FUNCTIONS
# ================================

def export_results_to_csv(final_rankings, feature_df, filename_prefix="ranking_results"):
    """Export results to CSV files for further analysis"""
    print(f"\n📁 Exporting results to CSV files...")

    try:
        # Export feature data
        feature_df.to_csv(f"{filename_prefix}_features.csv", index=False, encoding='utf-8-sig')
        print(f"   ✅ Features exported to {filename_prefix}_features.csv")

        # Export rankings for each user
        for user_id, rankings in final_rankings.items():
            rankings_df = pd.DataFrame(rankings)
            rankings_df.to_csv(f"{filename_prefix}_user_{user_id}.csv", index=False, encoding='utf-8-sig')
            print(f"   ✅ User {user_id} rankings exported to {filename_prefix}_user_{user_id}.csv")

    except Exception as e:
        print(f"   ❌ Export error: {str(e)}")


# ================================
# MAIN EXECUTION
# ================================

def main_custom():
    """Main execution function for customized Arabic data."""
    print("🚀 PERSONALIZED REVIEW RANKING SYSTEM")
    print("📊 Customized for Arabic Data with English Labels")
    print("=" * 70)

    reviews_df = load_your_data()
    user_queries = create_user_profiles_for_your_data()

    print(f"✅ Loaded {len(reviews_df)} reviews from your dataset")
    print(f"✅ Created {len(user_queries)} user profiles for testing\n")

    print("📋 Sample of Your Raw Data:")
    # Display the correct review column 'reviewText_ar'
    print(reviews_df[['index', 'asin', 'reviewText_ar']].head())
    print(f"\nDataset shape: {reviews_df.shape}")
    print(f"Columns: {list(reviews_df.columns)}\n")

    # Phase 1: Feature Engineering
    try:
        feature_df = phase1_feature_engineering_custom(reviews_df)
        print(f"✅ Phase 1 Complete: Extracted features for {len(feature_df)} reviews\n")

        print("📊 Feature Engineering Summary:")
        print(f"   - Text features: {len([col for col in feature_df.columns if col.startswith('text_')])} metrics")
        print(f"   - Domain features: {len([col for col in feature_df.columns if 'mentions' in col])} domain-specific signals")
        print(f"   - Sentiment features: {len([col for col in feature_df.columns if col.startswith('sentiment_')])} sentiment metrics")
        print(f"   - Average quality score: {feature_df['quality_score'].mean():.2f}\n")

        # Plot the distribution of quality scores
        plot_quality_scores(feature_df)

    except Exception as e:
        print(f"❌ Error in Phase 1: {str(e)}")
        return

    # Phase 2: Personalization
    try:
        personalized_results = phase2_personalization_custom(feature_df, user_queries)
        print(f"✅ Phase 2 Complete: Personalized rankings for {len(personalized_results)} users\n")
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

        # Call classification performance metrics for precision, recall, F1 score
        evaluate_classification_performance(feature_df)

    except Exception as e:
        print(f"❌ Error in Evaluation: {str(e)}")

    print("\n" + "=" * 70)
    print("🎉 SYSTEM EXECUTION COMPLETE!")
    print("=" * 70)

    return final_rankings, feature_df, personalized_results


from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_classification_performance(df):
    """Evaluate precision, recall, and F1 score for classification task"""
    
    # Let's define 'good' reviews as those with rating >= 4
    y_true = (df['rating'] >= 4).astype(int)  # Actual labels (1 if rating >= 4, else 0)
    
    # We will predict good reviews based on the review's quality score
    y_pred = (df['quality_score'] >= 6).astype(int)  # Predicted labels (1 if quality_score >= 6, else 0)

    # Calculate Precision, Recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Print the evaluation metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def plot_quality_scores(feature_df):
    """Plot distribution of the quality scores"""
    plt.figure(figsize=(8, 6))
    plt.hist(feature_df['quality_score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Quality Scores')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # The script will now run the entire pipeline with your specified settings
    final_rankings, feature_df, personalized_results = main_custom()
    export_results_to_csv(final_rankings, feature_df)
    evaluate_classification_performance(feature_df)
    plot_quality_scores(feature_df)
