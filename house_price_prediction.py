"""
Kaggle House Price Prediction using Linear Regression
===================================================

This script builds a Linear Regression model to predict house prices using the 
Kaggle House Prices dataset, focusing on three key features:
- GrLivArea (Above ground living area square feet) → square_feet
- BedroomAbvGr (Number of bedrooms above basement level) → bedrooms
- FullBath + HalfBath (Total bathrooms) → bathrooms

Steps:
1. Load Kaggle dataset (train.csv)
2. Feature engineering and selection
3. Handle missing values
4. Train/test split
5. Train LinearRegression model
6. Evaluate using RMSE and R² score
7. Visualize results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import subprocess
import platform
import warnings
warnings.filterwarnings('ignore')

print("Starting script...")
print("All imports successful!")

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    print("Plot style configured")
except Exception as e:
    print(f"Plot style warning: {e}")
    plt.style.use('default')
    print("Using default plot style")

def load_dataset():
    """
    Load the Kaggle House Prices dataset
    """
    print("=" * 60)
    print("LOADING KAGGLE DATASET")
    print("=" * 60)
    
    # Check if files exist
    if os.path.exists('train.csv'):
        train_df = pd.read_csv('train.csv')
        print(f"Loaded Kaggle dataset with {len(train_df)} samples")
    else:
        print("train.csv not found!")
        print("Please download the Kaggle House Prices dataset:")
        print("   1. Go to https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
        print("   2. Download train.csv")
        print("   3. Place it in the current directory")
        return None
    
    print(f"Dataset shape: {train_df.shape}")
    print(f"Available columns: {list(train_df.columns)}")
    
    return train_df

def prepare_features(df):
    """
    Extract and prepare the three main features for house price prediction
    """
    print("\n" + "=" * 60)
    print("FEATURE PREPARATION")
    print("=" * 60)
    
    # Define our target features mapping
    required_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None
    
    # Create new dataframe with our target features
    model_df = pd.DataFrame()
    
    # Map Kaggle features to our standard names
    model_df['square_feet'] = df['GrLivArea']  # Above ground living area
    model_df['bedrooms'] = df['BedroomAbvGr']  # Bedrooms above basement
    model_df['bathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5  # Total bathrooms
    model_df['price'] = df['SalePrice']  # Target variable
    
    print("Feature mapping completed:")
    print("  - square_feet: GrLivArea")
    print("  - bedrooms: BedroomAbvGr") 
    print("  - bathrooms: FullBath + HalfBath*0.5")
    print("  - price: SalePrice")
    
    # Handle missing values
    initial_missing = model_df.isnull().sum().sum()
    print(f"\nMissing values: {initial_missing}")
    
    if initial_missing > 0:
        # Fill missing values with median for numeric columns
        for col in model_df.columns:
            if model_df[col].isnull().sum() > 0:
                model_df[col].fillna(model_df[col].median(), inplace=True)
                print(f"  - Filled {col} missing values with median")
    
    print(f"\nFinal dataset shape: {model_df.shape}")
    print("\nDataset summary:")
    print(model_df.describe())
    
    print("\nCorrelation with price:")
    correlations = model_df.corr()['price'].drop('price').sort_values(ascending=False)
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.3f}")
    
    return model_df

def train_model(df):
    """
    Train Linear Regression model and evaluate performance
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Prepare features and target
    X = df[['square_feet', 'bedrooms', 'bathrooms']]
    y = df['price']
    
    print(f"Features: {list(X.columns)}")
    print(f"Training samples: {len(X)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Testing RMSE: ${test_rmse:,.2f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    
    # Model coefficients
    print(f"\nModel Coefficients:")
    feature_names = ['square_feet', 'bedrooms', 'bathrooms']
    for feature, coef in zip(feature_names, model.coef_):
        print(f"  {feature}: ${coef:,.2f}")
    print(f"  Intercept: ${model.intercept_:,.2f}")
    
    return model, X_test, y_test, y_test_pred, test_rmse, test_r2

def create_visualizations(df, model, X_test, y_test, y_test_pred, test_rmse, test_r2):
    """
    Create visualizations for the analysis
    """
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set font sizes for better readability
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Kaggle House Prices - Linear Regression Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Correlation heatmap
    ax1 = plt.subplot(2, 4, 1)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=ax1, fmt='.2f', annot_kws={'size': 8})
    ax1.set_title('Feature Correlation Matrix', pad=20)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)
    
    # 2. Price distribution
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_title('Distribution of House Prices', pad=20)
    ax2.set_xlabel('Sale Price ($)', labelpad=10)
    ax2.set_ylabel('Frequency', labelpad=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 3. Square feet vs Price
    ax3 = plt.subplot(2, 4, 3)
    ax3.scatter(df['square_feet'], df['price'], alpha=0.6, color='coral', s=20)
    ax3.set_title('Square Feet vs Price', pad=20)
    ax3.set_xlabel('Living Area (sq ft)', labelpad=10)
    ax3.set_ylabel('Sale Price ($)', labelpad=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 4. Bedrooms vs Price
    ax4 = plt.subplot(2, 4, 4)
    sns.boxplot(x='bedrooms', y='price', data=df, ax=ax4)
    ax4.set_title('Bedrooms vs Price', pad=20)
    ax4.set_xlabel('Number of Bedrooms', labelpad=10)
    ax4.set_ylabel('Sale Price ($)', labelpad=10)
    ax4.tick_params(axis='x', labelsize=8)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 5. Bathrooms vs Price
    ax5 = plt.subplot(2, 4, 5)
    sns.boxplot(x='bathrooms', y='price', data=df, ax=ax5)
    ax5.set_title('Bathrooms vs Price', pad=20)
    ax5.set_xlabel('Number of Bathrooms', labelpad=10)
    ax5.set_ylabel('Sale Price ($)', labelpad=10)
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 6. Actual vs Predicted
    ax6 = plt.subplot(2, 4, 6)
    ax6.scatter(y_test, y_test_pred, alpha=0.6, color='green', s=20)
    ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax6.set_title(f'Actual vs Predicted Prices\nRMSE: ${test_rmse:,.0f}, R²: {test_r2:.3f}', pad=20)
    ax6.set_xlabel('Actual Price ($)', labelpad=10)
    ax6.set_ylabel('Predicted Price ($)', labelpad=10)
    ax6.tick_params(axis='x', rotation=45, labelsize=8)
    ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 7. Residuals plot
    ax7 = plt.subplot(2, 4, 7)
    residuals = y_test - y_test_pred
    ax7.scatter(y_test_pred, residuals, alpha=0.6, color='purple', s=20)
    ax7.axhline(y=0, color='r', linestyle='--')
    ax7.set_title('Residual Plot', pad=20)
    ax7.set_xlabel('Predicted Price ($)', labelpad=10)
    ax7.set_ylabel('Residuals ($)', labelpad=10)
    ax7.tick_params(axis='x', rotation=45, labelsize=8)
    ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 8. Feature importance
    ax8 = plt.subplot(2, 4, 8)
    feature_names = ['Sq Feet', 'Bedrooms', 'Bathrooms']
    coefficients = model.coef_
    
    bars = ax8.bar(feature_names, coefficients, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax8.set_title('Feature Coefficients', pad=20)
    ax8.set_ylabel('Coefficient Value ($)', labelpad=10)
    ax8.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Add value labels on bars
    for bar, coef in zip(bars, coefficients):
        height = bar.get_height()
        label_y = height + (max(coefficients) - min(coefficients)) * 0.02
        ax8.text(bar.get_x() + bar.get_width()/2, label_y,
                f'${coef:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.92, bottom=0.12,
        wspace=0.35, hspace=0.45
    )
    
    plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Plot saved successfully!")
    
    # Don't show plot in non-interactive mode, just save it
    plt.close()
    
    print("Visualizations saved as 'house_price_analysis.png'")

def open_image(filename):
    """
    Open the generated image file using the default system viewer
    """
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', filename])
        elif platform.system() == 'Windows':  # Windows
            os.startfile(filename)
        else:  # Linux and other Unix-like systems
            subprocess.call(['xdg-open', filename])
        print(f"Opening {filename} in default image viewer...")
    except Exception as e:
        print(f"Could not open image automatically: {e}")
        print(f"Please manually open {filename} to view the visualizations")

def make_predictions(model):
    """
    Make example predictions
    """
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Example houses
    examples = [
        [1500, 3, 2.0],  # Average house
        [2500, 4, 3.0],  # Larger house  
        [1000, 2, 1.0],  # Smaller house
        [3000, 5, 4.0],  # Luxury house
    ]
    
    descriptions = [
        "Average House",
        "Larger House", 
        "Smaller House",
        "Luxury House"
    ]
    
    for example, desc in zip(examples, descriptions):
        prediction = model.predict([example])[0]
        print(f"\n{desc}:")
        print(f"  - Square Feet: {example[0]:,}")
        print(f"  - Bedrooms: {example[1]}")
        print(f"  - Bathrooms: {example[2]}")
        print(f"  - Predicted Price: ${prediction:,.2f}")

def main():
    """
    Main function to run the complete analysis
    """
    try:
        print("KAGGLE HOUSE PRICES - LINEAR REGRESSION ANALYSIS")
        print("=" * 70)
        
        # Step 1: Load dataset
        print("Step 1: Loading dataset...")
        df = load_dataset()
        if df is None:
            print("Failed to load dataset")
            return
        
        # Step 2: Prepare features
        print("Step 2: Preparing features...")
        model_df = prepare_features(df)
        if model_df is None:
            print("Failed to prepare features")
            return
        
        # Step 3: Train model and evaluate
        print("Step 3: Training model...")
        model, X_test, y_test, y_test_pred, test_rmse, test_r2 = train_model(model_df)
        
        # Step 4: Create visualizations
        print("Step 4: Creating visualizations...")
        create_visualizations(model_df, model, X_test, y_test, y_test_pred, test_rmse, test_r2)
        
        # Step 5: Make example predictions
        print("Step 5: Making predictions...")
        make_predictions(model)
        
        print(f"\nAnalysis Complete!")
        print(f"Check 'house_price_analysis.png' for visualizations")
        print(f"Model achieved R² score of {test_r2:.3f} with RMSE of ${test_rmse:,.0f}")
        
        # Open the generated visualization
        print("\nOpening visualization...")
        open_image('house_price_analysis.png')
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Script started!")
    main()
    print("Script finished!")
