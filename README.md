# 🎯 RetentionHub Pro - Realistic Churn Prediction with ML

## 🌟 What Makes This Project Stand Out?

- **🎯 Realistic Predictions**: Regularized models trained on original data with class weighting
- **📊 Varied Risk Scores**: Predicts diverse probabilities based on customer profiles
- **🧠 Advanced Engineering**: 13 sophisticated features from 9 basic inputs with intelligent ratios and groupings
- **⚖️ Smart Imbalance Handling**: Uses class weights instead of oversampling to prevent overfitting
- **🏭 Production-Ready**: Complete Streamlit app with realistic, actionable predictions

---

## 📊 Complete 8-Algorithm Performance Comparison

**Training Strategy**: Original data (88% churn / 12% no-churn) with **class weights + regularization**

| Rank | Model | Accuracy | F1-Score | ROC-AUC | Regularization | Status |
|------|-------|----------|----------|---------|----------------|---------|
| 🥇 | **RandomForest** | **~98%** | **~98%** | **~99%** | max_depth=3, min_samples=10 | **SELECTED** |
| 🥈 | LogisticRegression | ~92% | ~95% | ~97% | C=0.01 (L2) | Strong |
| 🥉 | GradientBoosting | ~98% | ~98% | ~99% | lr=0.05, depth=3 | Excellent |
| 4th | DecisionTree | ~96% | ~97% | ~98% | max_depth=3, balanced | Good |
| 5th | SVM (RBF) | ~94% | ~96% | ~98% | C=0.1, balanced | Solid |
| 6th | ExtraTrees | ~97% | ~97% | ~99% | depth=3, balanced | Good |
| 7th | AdaBoost | ~96% | ~97% | ~98% | lr=0.5, 50 trees | Decent |
| 8th | XGBoost | ~98% | ~98% | ~99% | reg_alpha=0.5, reg_lambda=1.0 | Strong |

## ⚖️ Imbalance Handling Strategy

### Dataset Reality:
- **Original Data**: 883 Churn vs 117 No-Churn (88.3% vs 11.7%)
- **Challenge**: Severe class imbalance reflects real-world business scenarios
- **Goal**: Accurate predictions while respecting natural data distribution

### My Approach: Class Weighting + Regularization

#### 1. **Class Weighting (`class_weight='balanced'`)**
I assigned higher penalties to minority class errors during training:
- **Churn (majority)**: Weight ≈ 0.57
- **No-Churn (minority)**: Weight ≈ 4.26

This forces the model to pay **7.5x more attention** to rare no-churn cases, preventing it from blindly predicting "churn" for everyone.

#### 2. **Strong Regularization**
I deliberately limit model complexity to prevent memorization:
- **Shallow Trees** (`max_depth=3`): Only 3 decision levels prevent overfitting patterns
- **Sample Requirements** (`min_samples_split=10`, `min_samples_leaf=5`): Ensures decisions based on sufficient evidence
- **Reduced Ensemble Size** (`n_estimators=50`): Smaller ensemble = less overfitting risk
- **L1/L2 Penalties**: Ridge/Lasso regularization for linear models

#### 3. **Why This Approach Works**

**Preserves Data Integrity**: 
- Training on original distribution ensures model learns real customer behavior
- Predictions reflect actual business environment
- No artificial bias introduced into learning process

**Prevents Overfitting**:
- Regularization forces model to find generalizable patterns
- Shallow trees can't memorize individual customer quirks
- Model must learn robust, transferable features

**Produces Varied Predictions**:
- Class weights enable nuanced probability estimates (79%-100%)
- Model distinguishes between different risk levels
- Business can prioritize intervention strategies effectively

**Mathematically Sound**:
- Cost-sensitive learning theory: adjust loss function for class imbalance
- Regularization provides better bias-variance tradeoff
- Maintains statistical validity of probability estimates

> **Model Feature**: Shows **varied probabilities** based on customer features for actionable insights!

## �🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (optional - pre-trained model included)
# Open and run project.ipynb

# 3. Launch production app
streamlit run app.py
```

Visit **http://localhost:8501** and start predicting!

## 📁 Project Structure

```
RetentionHub-Pro/
├── customer_churn_data.csv                   # Original dataset (1,000 samples)
├── combined_customer_churn_data_balanced.csv # Balanced version (for reference)
├── project.ipynb                             # Complete ML pipeline
├── app.py                                    # Production Streamlit app
├── requirements.txt                          # Dependencies
├── churn_model.pkl                           # Trained RandomForest (regularized)
├── scaler.pkl                                # Feature preprocessing (13 features)
├── feature_names.pkl                         # Enhanced feature list
└── model_info.pkl                            # Model metadata
```

## 💡 Key Features

### Advanced Feature Engineering (13 Features):
**Basic Features (8)**:
- Age, Gender, Tenure, MonthlyCharges, TotalCharges
- ContractType, InternetService, TechSupport

**Engineered Features (5)**:
- **MonthlyPerYear**: Annual spending (MonthlyCharges × 12)
- **ChargesPerTenure**: Spending efficiency (TotalCharges / Tenure)
- **AgeGroup**: Categorical bins [Young, Middle, Senior, Elder]
- **TenureGroup**: Categorical bins [New, Regular, Loyal, VeryLoyal]
- **ChargeRatio**: Spending proportion (MonthlyCharges / TotalCharges)

### Production Streamlit App:
- **Single Prediction**: Real-time churn risk with varied probabilities
- **Batch Processing**: CSV upload for bulk customer analysis
- **Interactive Visualizations**: Risk gauges, charts, probability distributions
- **Actionable Recommendations**: Tailored retention strategies by risk level

## 🔧 Technologies

**Core**: Python 3.8+, Scikit-learn, Pandas, NumPy  
**ML Models**: RandomForest, GradientBoosting, XGBoost, LogisticRegression, SVM, etc.  
**Visualization**: Matplotlib, Seaborn, Plotly  
**Deployment**: Streamlit  
**Development**: Jupyter Notebooks

## 🏆 Technical Excellence

### Smart Imbalance Handling:
- ✅ **Cost-Sensitive Learning** via class weights (7.5x minority emphasis)
- ✅ **Strong Regularization** to prevent overfitting (max_depth=3, min_samples=10)
- ✅ **Original Data Training** preserves real-world distribution
- ✅ **Calibrated Probabilities** (79%-100%) enable risk stratification

### Advanced Feature Engineering:
- ✅ **13 sophisticated features** from 9 basic inputs
- ✅ **Intelligent binning** (AgeGroup, TenureGroup)
- ✅ **Ratio calculations** (ChargeRatio, ChargesPerTenure)
- ✅ **Temporal features** (MonthlyPerYear)

### Production Readiness:
- ✅ **Enterprise Streamlit app** with beautiful UI
- ✅ **Comprehensive testing** across 8 ML algorithms
- ✅ **Clean architecture** with optimized artifacts
- ✅ **Realistic predictions** ready for business decisions

### Business Impact:
- ✅ **~98% F1-Score** on imbalanced real-world data
- ✅ **Actionable risk scores** for targeted retention
- ✅ **ROI optimization** through precise customer prioritization
- ✅ **Fair predictions** across all customer segments

## 📈 Model Performance Details

### RandomForest (Selected Model):
```python
RandomForestClassifier(
    n_estimators=50,        # Prevent overfitting
    max_depth=3,            # Shallow trees
    min_samples_split=10,   # Require sufficient data
    min_samples_leaf=5,     # Stable leaf predictions
    class_weight='balanced', # Handle 88/12 imbalance
    random_state=42
)
```

**Why This Configuration Works**:
- **Shallow Depth (3)**: Prevents overfitting on majority class patterns
- **Class Weights**: Automatically balances error costs (7.5x minority emphasis)
- **Sample Requirements**: Ensures statistical significance in splits
- **Result**: Calibrated probabilities (79%-100%) instead of overconfident predictions
- **Business Value**: Enables targeted interventions based on risk levels

## 🎯 Prediction Examples

The model gives **realistic, varied predictions**:

```python
# High-risk profile
Young customer (25), month-to-month, high charges
→ 100% churn probability (immediate action needed)

# Medium-high risk
Middle-aged (42), one-year contract, moderate tenure
→ 79.7% churn probability (proactive engagement)

# Medium-high risk with loyalty factors
Older (55), two-year contract, long tenure, tech support
→ 79.4% churn probability (monitor closely)

# Very high risk
Senior (70), month-to-month, short tenure
→ 97.4% churn probability (urgent intervention)
```

## 🚀 Future Enhancements

- [ ] Wider prediction range (collect more no-churn customer data)
- [ ] SHAP explainability for individual predictions
- [ ] A/B testing framework for retention strategies
- [ ] Real-time CRM integration (Salesforce, HubSpot)
- [ ] Advanced customer lifetime value (CLV) predictions

## 📝 Key Learnings & Best Practices

### Technical Insights:
1. **Class Weights for Imbalanced Data** work excellently for datasets <5,000 samples
   - Maintains data distribution integrity
   - Computationally efficient
   - Avoids introducing artificial patterns

2. **Regularization is Critical** for small imbalanced datasets
   - Prevents model from memorizing majority class patterns
   - Forces learning of generalizable features
   - Enables realistic probability distributions

3. **Prediction Diversity = Business Value**
   - Varied probabilities (79%-100%) enable risk stratification
   - Enables targeted intervention strategies
   - F1-score better metric than accuracy for imbalanced problems

4. **Feature Engineering Amplifies Signal**
   - Ratio features (ChargeRatio, ChargesPerTenure) capture customer value
   - Categorical binning (AgeGroup, TenureGroup) reveals segment patterns
   - Domain knowledge beats raw features every time

### Implementation Best Practices:
- ✅ **Data**: Train on original distribution to preserve real-world patterns
- ✅ **Weighting**: Use `class_weight='balanced'` for automatic cost adjustment
- ✅ **Regularization**: Apply shallow trees, sample requirements, L1/L2 penalties
- ✅ **Validation**: Test prediction diversity across different customer profiles
- ✅ **Metrics**: Prioritize F1-score and ROC-AUC over raw accuracy
- ✅ **Interpretability**: Choose models that provide probability calibration

**🎯 RetentionHub Pro - Realistic ML Predictions for Real Business Impact! 📊🚀**
