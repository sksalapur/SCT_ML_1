# Kaggle House Price Prediction with Linear Regression

This project builds a Linear Regression model to predict house prices using the famous **Kaggle House Prices dataset**. The model focuses on three key features: square footage, number of bedrooms, and number of bathrooms.

## 📋 Project Overview

The model predicts house prices using:
- **Square footage** - Above ground living area (GrLivArea)
- **Number of bedrooms** - Bedrooms above basement level (BedroomAbvGr)
- **Number of bathrooms** - Total bathrooms (FullBath + HalfBath × 0.5)

## 🛠️ Technology Stack

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning library for Linear Regression
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **numpy** - Numerical computing

## 📊 Dataset

This project uses the **Kaggle House Prices: Advanced Regression Techniques** dataset.

### Required Dataset Files:
- `train.csv` - Training data with house features and sale prices

### How to Get the Dataset:
1. Go to [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Download `train.csv`
3. Place it in the project directory

### Key Features Used:
- `GrLivArea` → **square_feet** - Above ground living area
- `BedroomAbvGr` → **bedrooms** - Number of bedrooms above basement
- `FullBath` + `HalfBath` → **bathrooms** - Total bathrooms
- `SalePrice` → **price** - Target variable

## 🚀 Getting Started

### Prerequisites
- Python 3.x installed
- Kaggle dataset files (`train.csv`)

### Installation

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the Kaggle dataset and place `train.csv` in the project directory

### Running the Analysis

Execute the main script:
```bash
python house_price_prediction.py
```

## 📈 Analysis Steps

The script performs:

1. **Dataset Loading**
   - Loads `train.csv` from Kaggle dataset
   - Validates required columns are present

2. **Feature Engineering**
   - Maps Kaggle columns to standard feature names
   - Combines FullBath and HalfBath into total bathrooms
   - Handles missing values with median imputation

3. **Data Preprocessing**
   - Cleans and validates data
   - Creates feature correlation analysis

4. **Train/Test Split**
   - 80% training, 20% testing
   - Uses random_state=42 for reproducibility

5. **Model Training**
   - Trains Linear Regression model using sklearn
   - Calculates feature coefficients

6. **Model Evaluation**
   - RMSE (Root Mean Square Error)
   - R² Score (coefficient of determination)
   - Performance comparison between training and testing

7. **Visualization**
   - Feature correlation heatmap
   - Price distribution
   - Feature vs price relationships
   - Actual vs predicted comparison
   - Residual analysis
   - Feature importance visualization

## 📊 Expected Results

With the Kaggle dataset, the model typically achieves:
- **R² Score**: 0.60-0.80 (depending on data quality)
- **RMSE**: $30,000-50,000

### Feature Importance
- **Square footage** typically has the highest impact on price
- **Bedrooms** and **bathrooms** provide additional predictive value
- Linear relationship works well for basic house price prediction

## 📁 Output Files

- `house_price_analysis.png` - Comprehensive visualization dashboard
- Console output with detailed analysis results

## 🔍 Model Interpretation

### Example Prediction
For a house with:
- 2,000 square feet
- 3 bedrooms  
- 2 bathrooms

The model will predict a price based on learned coefficients from the Kaggle dataset.

## 🎯 Learning Objectives

This project demonstrates:
- Real-world dataset handling (Kaggle competition data)
- Feature engineering and selection
- Linear regression with scikit-learn
- Model evaluation metrics (RMSE, R²)
- Data visualization techniques
- Machine learning workflow best practices

## 📋 Project Structure

```
house-price-prediction/
├── house_price_prediction.py    # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                   # Project documentation
├── train.csv                   # Kaggle dataset (user downloads)
└── house_price_analysis.png    # Generated visualizations
```

## 🔧 Customization

You can modify the script to:
- Add more features from the Kaggle dataset
- Try different regression models (Ridge, Lasso, etc.)
- Experiment with feature scaling and normalization
- Implement cross-validation
- Add polynomial features for non-linear relationships

## 📚 Dataset Citation

This project uses the House Prices dataset from:
- **Competition**: House Prices: Advanced Regression Techniques
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.
