# Personalized Review Recommendation System

A machine learning-powered web application that provides personalized review recommendations for entertainment products (movies, TV series) using advanced feature engineering, semantic analysis, and Ridge regression.

## 🎯 Project Overview

This system analyzes user behavior patterns and preferences to recommend the most relevant product reviews, supporting both English and Arabic text. It's specifically designed for entertainment products with a focus on historical war series and premium content.

## 🚀 Features

### Core Functionality
- **Personalized Recommendations**: Tailored review suggestions based on user profiles and preferences
- **Multi-language Support**: English and Arabic review display
- **Advanced ML Ranking**: Ridge regression-based scoring system
- **Real-time Processing**: Dynamic recommendation generation
- **User Session Management**: Persistent user preferences

### Technical Features
- **Enhanced Feature Engineering**: 40+ extracted features including sentiment analysis, domain-specific keywords, and text quality metrics
- **Semantic Similarity**: TF-IDF based content matching
- **Quality Scoring**: Comprehensive review quality assessment
- **Temporal Analysis**: Recency-based review weighting

## 🛠️ Technology Stack

### Backend
- **Python 3.x**
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **NumPy** - Numerical computing

### Frontend
- **HTML/CSS/JavaScript**
- **Jinja2** - Template engine
- **Bootstrap** (assumed for styling)

### Machine Learning
- **Ridge Regression** - Primary ranking model
- **TF-IDF Vectorization** - Text similarity
- **Feature Engineering** - Custom domain-specific features
- **Cross-validation** - Model optimization

## 📁 Project Structure

```
recommendation-system/
│
├── app.py                      # Main Flask application
├── Movies_and_TV_Reviews.csv   # English reviews dataset
├── user_profiles.json          # User preference profiles
├── templates/
│   ├── home.html              # Product catalog page
│   └── product.html           # Product detail & recommendations
├── static/                    # CSS, JS, images
└── README.md                  # This file
```

## 📊 Dataset

### Products Catalog
- Game of Thrones ($99.99)
- Breaking Bad ($699.99)
- Lord of the Rings ($1,299.99)
- The Godfather ($14.99)
- The Hobbit ($9.99)
- House of the Dragon ($24.99)

### Review Features
- **Text Content**: Original review text in English/Arabic
- **Ratings**: 1-5 star ratings
- **Helpfulness**: Voting data
- **Verification**: Purchase verification status
- **Temporal**: Review timestamps
- **Product Mapping**: ASIN identifiers

## 🤖 Machine Learning Pipeline

### Phase 1: Feature Engineering
- **Text Features**: Length, word count, sentence structure, punctuation density
- **Domain Features**: War/history mentions, series quality, emotional impact
- **Sentiment Features**: Positive/negative word counts, intensity analysis
- **Quality Indicators**: Helpfulness ratio, verification status
- **Temporal Features**: Recency scoring, freshness indicators

### Phase 2: Personalization
- **Semantic Matching**: TF-IDF cosine similarity
- **Interest Alignment**: User preference matching
- **Demographic Targeting**: Age group considerations
- **Purchase History**: Context-based scoring

### Phase 3: ML Ranking
- **Ridge Regression**: Primary ranking algorithm
- **Feature Scaling**: StandardScaler normalization
- **Cross-validation**: Hyperparameter tuning
- **Composite Scoring**: Multi-factor relevance calculation

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.7+
pip package manager
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn flask requests datetime collections
```

### Required Files
1. `Movies_and_TV_Reviews.csv` - Review dataset
2. `user_profiles.json` - User preference profiles
3. Product images in `/static/images/`

### Run Application
```bash
python app.py
```

Access the application at `http://localhost:5000`

## 🎮 Usage

### For Users
1. **Browse Products**: Visit home page to see available items
2. **Select Product**: Click on any product for detailed view
3. **Set User Profile**: Use `/set-user/<user_id>` to switch users
4. **View Recommendations**: See top 3 personalized reviews
5. **Language Toggle**: Switch between English/Arabic with `/set-lang/<lang>`

### For Developers
```python
# Generate recommendations for specific user/product
recommendations = run_complete_recommendation_system_ridge_only(
    user_id=2, 
    asin='B00R8GUXPG'
)

# Access recommendation details
for rec in recommendations:
    print(f"Rank: {rec['rank']}")
    print(f"Review: {rec['text']}")
    print(f"Score: {rec['ridge_score']}")
```

## 📈 Performance Metrics

### Model Evaluation
- **Cross-validation Score**: Optimized through alpha tuning
- **Feature Importance**: Top features identified and ranked
- **Ranking Quality**: Multiple scoring approaches compared

### System Metrics
- **Processing Time**: Real-time recommendation generation
- **Memory Usage**: Efficient data handling with pandas
- **Scalability**: Supports multiple concurrent users

## 🎯 User Profiles

The system supports different user personas:

### Profile Examples
- **History Enthusiasts**: Focus on war/military content
- **Entertainment Seekers**: Emphasis on series quality
- **Gift Buyers**: Personal context and value assessment
- **Quality Seekers**: Premium content and production values

### Demographic Targeting
- Age groups: 35-50, 45-60
- Interests: History, entertainment, emotional content
- Purchase patterns: Gift giving, premium content

## 🔍 API Endpoints

### Web Routes
- `GET /` - Product catalog homepage
- `GET /product/<product_id>` - Product detail page with recommendations
- `GET /set-user/<user_id>` - Set active user session
- `GET /set-lang/<lang_code>` - Set display language (en/ar)

### Response Format
```json
{
    "rank": 1,
    "review_id": 12345,
    "ridge_score": 7.85,
    "rating": 5,
    "text": "Review content...",
    "explanation": "Recommended because...",
    "key_metrics": {
        "quality_score": 4.2,
        "helpfulness_ratio": 0.85,
        "semantic_similarity": 0.75
    }
}
```


