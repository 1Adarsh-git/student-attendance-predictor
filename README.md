# 📚 Student Event Attendance Predictor

**A beginner-friendly, end-to-end machine learning pipeline that predicts whether a student will attend a campus event using binary classification.**

🚀 **Perfect for hackathons, tech events, and showcasing your ML skills!**

## 📋 Project Overview

This project builds a complete ML pipeline to predict student event attendance based on:
- Student profile (department, year, interests, past attendance)
- Event characteristics (type, tags, location, timing)  
- Additional factors (notifications, distance, registration channel)

**Model Performance:** ROC AUC = 0.76 | F1 Score = 0.46 | Accuracy = 67%

## 🗂️ Project Structure

```
student-attendance-predictor/
├─ data/
│  ├─ events.csv                    # 200 synthetic events
│  ├─ users.csv                     # 1000 synthetic students  
│  ├─ attendance.csv                # 5000 user-event pairs with labels
│  └─ processed_data.npz           # Preprocessed training data
├─ notebooks/
│  ├─ 01_data_generation_and_EDA.ipynb     # Data generation & analysis
│  └─ 02_modeling_and_evaluation.ipynb    # Model training & evaluation
├─ src/
│  ├─ data_generation.py           # Generate synthetic datasets
│  ├─ preprocess.py                # Feature engineering pipeline
│  ├─ model_train.py               # Model training & evaluation
│  ├─ explainability.py            # Model interpretation
│  └─ app_streamlit.py             # Interactive web demo
├─ models/
│  ├─ attendance_model.pkl         # Best trained model
│  └─ preprocessor.joblib          # Fitted preprocessing pipeline
├─ outputs/
│  ├─ figures/                     # All visualization outputs
│  ├─ metrics_summary.csv          # Model performance metrics
│  └─ feature_importance.csv       # Feature importance rankings
├─ requirements.txt                # Python dependencies
├─ README.md                       # This file
└─ demo_script.txt                # 90-second presentation script
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd student-attendance-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Demo

```bash
# Launch the Streamlit web app
streamlit run src/app_streamlit.py
```

The demo will open in your browser with an interactive interface to:
- Input student and event characteristics
- Get real-time attendance predictions with explanations
- Upload CSV files for batch predictions

### 3. Explore the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open either notebook:
# - notebooks/01_data_generation_and_EDA.ipynb
# - notebooks/02_modeling_and_evaluation.ipynb
```

### 4. Regenerate Everything from Scratch

```bash
# Generate new synthetic data
python src/data_generation.py

# Preprocess the data
python src/preprocess.py

# Train models
python src/model_train.py

# Generate explainability reports
python src/explainability.py
```

## 📊 Data Schema

### Users (1000 students)
- **user_id**: Unique identifier (u1-u1000)
- **dept**: Department (CSE, ECE, ME, CE, EE, IT)  
- **year**: Academic year (1-4)
- **interests**: Comma-separated tags (ML,AI,Web,Finance,etc.)
- **past_attendance_count**: Historical event attendance (0-20)

### Events (200 events) 
- **event_id**: Unique identifier (e1-e200)
- **title**: Event name
- **event_type**: Workshop, Talk, Hackathon, Competition
- **tags**: Comma-separated interest tags
- **day_of_week**: Mon-Sun
- **time_of_day**: Morning/Afternoon/Evening

### Attendance (5000 records)
- All user and event features combined
- **notification_received**: Whether student was notified (0/1)
- **distance_km**: Distance to venue (0-30 km)
- **register_channel**: email, whatsapp, instagram, none
- **attend**: TARGET VARIABLE (0=No, 1=Yes) - 19.3% attendance rate

## 🤖 Modeling Approach

### Feature Engineering
- **Categorical encoding**: One-hot encoding for departments, event types, etc.
- **Tag processing**: Multi-hot encoding for interest/event tag matching
- **Derived features**: Tag match count, scaled numerical features
- **Final feature count**: 40 features

### Models Trained
1. **Logistic Regression** (Winner: ROC AUC = 0.76)
   - Hyperparameters: C=1.0, class_weight='balanced', solver='liblinear'
   - Strengths: Fast, interpretable, handles class imbalance well

2. **Random Forest** (ROC AUC = 0.74)  
   - Hyperparameters: 100 estimators, max_depth=10, class_weight='balanced'
   - Strengths: Robust, feature importance, non-linear patterns

### Key Predictive Features
1. **Notification received** - Critical for attendance (+20% probability)
2. **Tag match count** - Interest alignment drives engagement  
3. **Past attendance** - Historical behavior predicts future
4. **Distance from venue** - Geographic accessibility matters
5. **Event type** - Workshops/Hackathons > Talks

## 📈 Model Performance

| Model | ROC AUC | F1 Score | Accuracy | Precision (Attend) | Recall (Attend) |
|-------|---------|----------|----------|-------------------|----------------|
| **Logistic Regression** | **0.760** | **0.459** | **67.0%** | **33.6%** | **72.5%** |
| Random Forest | 0.735 | 0.384 | 78.5% | 43.0% | 34.7% |

**Interpretation**: The logistic regression model excels at identifying students likely to attend (high recall), making it ideal for event planning and resource allocation.

## 🧠 Business Insights

### For Event Organizers:
- **📲 Prioritize notifications**: 70% improvement in attendance with notifications
- **🎯 Target relevant audiences**: Match events to student interests (+12% per matching tag)
- **📍 Choose accessible venues**: Each km distance reduces attendance by 1%
- **👥 Engage past attendees**: Students with 5+ past events are 10% more likely to attend
- **🎪 Format matters**: Workshops and Hackathons outperform traditional talks

## 🔧 Technical Features

- **Reproducible**: All random operations use `random_state=42`
- **Modular design**: Clean separation between data, preprocessing, training, and inference
- **Production-ready**: Saved models and preprocessors for deployment
- **Well-documented**: Comprehensive notebooks and inline documentation
- **Beginner-friendly**: Uses only standard ML libraries (scikit-learn, pandas, streamlit)

## 📱 Streamlit Demo Features

- **Interactive prediction**: Real-time attendance probability calculation
- **Smart explanations**: Human-readable reasons for each prediction
- **Batch processing**: Upload CSV for multiple predictions
- **Visual insights**: Department and event type statistics
- **User-friendly**: Clean interface with helpful tooltips

## 🎯 Use Cases

### For Students (like you!):
- **Hackathon projects**: Complete end-to-end ML pipeline
- **Portfolio showcase**: Demonstrates data science skills
- **Interview preparation**: Real-world problem with business impact
- **Learning tool**: Best practices for ML project structure

### For Institutions:
- **Event planning**: Optimize resource allocation
- **Marketing strategy**: Target high-probability attendees  
- **Venue selection**: Data-driven location decisions
- **Performance tracking**: Monitor event success factors

## 🚀 Future Improvements

### Model Enhancements:
- **Advanced algorithms**: Try XGBoost, LightGBM for better performance
- **Deep learning**: Neural networks for complex pattern recognition
- **Ensemble methods**: Combine multiple models for robust predictions

### Feature Engineering:
- **Temporal features**: Seasonality, exam periods, holidays
- **Social features**: Friends attending, peer influence
- **Weather data**: Impact of weather conditions
- **Historical success**: Event-specific attendance patterns

### Production Readiness:
- **Real-time inference API**: FastAPI deployment
- **Model monitoring**: Performance tracking over time
- **A/B testing**: Compare model versions
- **Feedback loop**: Continuous learning from new data

## 🤝 Contributing

This project is designed for learning and experimentation. Feel free to:
- 🔧 Modify the synthetic data generation logic
- 🤖 Try different ML algorithms  
- 📊 Add new visualizations
- 🎨 Enhance the Streamlit UI
- 📝 Improve documentation

## 📄 License

This project is for educational purposes. Feel free to use it in your portfolio, hackathons, or learning journey!

## 👨‍💻 Author

**Built as a comprehensive ML learning project - perfect for students entering the tech industry!**

---

### 💡 Pro Tips for Presentations:

1. **Start with the business problem**: "How can we predict student event attendance?"
2. **Show the data**: Real examples from the synthetic dataset
3. **Demo the app**: Live predictions with explanations
4. **Discuss insights**: What factors drive attendance?
5. **Technical highlights**: Feature engineering, model selection, evaluation metrics

**Perfect for:** Hackathons • Internship applications • Portfolio projects • Technical interviews

---

*This project demonstrates end-to-end machine learning skills highly valued by tech companies and startups. Use it to showcase your ability to solve real-world problems with data!*
