import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="PropertyPredict Pro",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the Kaggle house price data"""
    try:
        # Load the dataset
        df = pd.read_csv('train.csv')
        
        # Select required columns
        required_columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
        df_clean = df[required_columns].copy()
        
        # Handle missing values
        df_clean = df_clean.dropna()
        
        # Create feature mapping
        df_clean['square_feet'] = df_clean['GrLivArea']
        df_clean['bedrooms'] = df_clean['BedroomAbvGr']
        df_clean['bathrooms'] = df_clean['FullBath'] + df_clean['HalfBath'] * 0.5
        df_clean['price'] = df_clean['SalePrice']
        
        # Select final features
        features = ['square_feet', 'bedrooms', 'bathrooms', 'price']
        return df_clean[features]
    
    except FileNotFoundError:
        st.error("❌ train.csv file not found! Please upload the Kaggle dataset.")
        return None

@st.cache_data
def train_model(df):
    """Train the linear regression model"""
    X = df[['square_feet', 'bedrooms', 'bathrooms']]
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    return model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

def predict_price(model, square_feet, bedrooms, bathrooms):
    """Predict house price using the trained model"""
    features = np.array([[square_feet, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    return prediction

def usd_to_inr(usd_amount, exchange_rate=83.50):
    """Convert USD to INR using current exchange rate"""
    return usd_amount * exchange_rate

def create_feature_importance_chart(model, currency="USD"):
    """Create feature importance visualization"""
    features = ['Square Feet', 'Bedrooms', 'Bathrooms']
    coefficients = model.coef_
    
    if currency == "INR":
        coefficients_display = [usd_to_inr(coef) for coef in coefficients]
        text_values = [f'₹{coef:,.0f}' for coef in coefficients_display]
        y_title = "Coefficient Value (₹)"
    else:
        coefficients_display = coefficients
        text_values = [f'${coef:,.0f}' for coef in coefficients_display]
        y_title = "Coefficient Value ($)"
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=coefficients_display,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=text_values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Feature Importance (Coefficients) - {currency}",
        xaxis_title="Features",
        yaxis_title=y_title,
        template="plotly_white",
        height=400
    )
    
    return fig

def create_prediction_vs_actual_chart(y_test, y_pred_test, currency="USD"):
    """Create prediction vs actual price chart"""
    fig = go.Figure()
    
    if currency == "INR":
        y_test_display = [usd_to_inr(price) for price in y_test]
        y_pred_display = [usd_to_inr(price) for price in y_pred_test]
        x_title = "Actual Price (₹)"
        y_title = "Predicted Price (₹)"
    else:
        y_test_display = y_test
        y_pred_display = y_pred_test
        x_title = "Actual Price ($)"
        y_title = "Predicted Price ($)"
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=y_test_display,
        y=y_pred_display,
        mode='markers',
        marker=dict(
            color='rgba(67, 147, 195, 0.7)',
            size=8,
            line=dict(width=1, color='white')
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test_display), min(y_pred_display))
    max_val = max(max(y_test_display), max(y_pred_display))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=f"Predicted vs Actual Prices - {currency}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=400
    )
    
    return fig

def create_price_distribution_chart(df, currency="USD"):
    """Create price distribution chart"""
    if currency == "INR":
        price_data = [usd_to_inr(price) for price in df['price']]
        x_title = "Price (₹)"
        title = "House Price Distribution - INR"
    else:
        price_data = df['price']
        x_title = "Price ($)"
        title = "House Price Distribution - USD"
    
    fig = px.histogram(
        x=price_data, 
        nbins=50,
        title=title,
        color_discrete_sequence=['#8B5CF6']
    )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏡 PropertyPredict Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Real Estate Valuation • Instant Property Insights</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_and_prepare_data()
    
    if df is None:
        st.stop()
    
    # Train model
    model, metrics = train_model(df)
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">🎛️ Property Details</h2>', unsafe_allow_html=True)
    
    # Currency selection
    currency = st.sidebar.radio(
        "💱 Display Currency",
        options=["USD ($)", "INR (₹)", "Both"],
        index=2,
        help="Choose your preferred currency display"
    )
    
    # Input widgets
    square_feet = st.sidebar.number_input(
        "🏡 Square Feet",
        min_value=int(df['square_feet'].min()),
        max_value=int(df['square_feet'].max()),
        value=int(df['square_feet'].median()),
        step=50,
        help="Above ground living area in square feet"
    )
    
    bedrooms = st.sidebar.selectbox(
        "🛏️ Number of Bedrooms",
        options=sorted(df['bedrooms'].unique()),
        index=2,
        help="Number of bedrooms above basement level"
    )
    
    # Convert bathrooms to whole numbers only
    bathroom_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    bathrooms = st.sidebar.selectbox(
        "🚿 Number of Bathrooms",
        options=bathroom_options,
        index=3,
        help="Total bathrooms (0.5 = half bath, 1.0 = full bath, etc.)"
    )
    
    # Bathroom explanation
    with st.sidebar.expander("ℹ️ Bathroom Count Explanation"):
        st.markdown("""
        **How bathroom counting works:**
        - **0.5** = Half bathroom (toilet + sink only)
        - **1.0** = Full bathroom (toilet + sink + shower/tub)
        - **1.5** = 1 full + 1 half bathroom
        - **2.0** = 2 full bathrooms
        - And so on...
        """)
    
    # Predict button
    if st.sidebar.button("🔮 Get Property Valuation", type="primary"):
        st.session_state.prediction_made = True
        st.session_state.predicted_price = predict_price(model, square_feet, bedrooms, bathrooms)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction result
        if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
            predicted_price = st.session_state.predicted_price
            predicted_price_inr = usd_to_inr(predicted_price)
            
            # Display based on currency selection
            if currency == "USD ($)":
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Property Valuation</h2>
                    <h1>${predicted_price:,.0f} USD</h1>
                    <p>For {square_feet:,} sq ft • {bedrooms} bed • {bathrooms} bath</p>
                </div>
                """, unsafe_allow_html=True)
            elif currency == "INR (₹)":
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Property Valuation</h2>
                    <h1>₹{predicted_price_inr:,.0f} INR</h1>
                    <p>For {square_feet:,} sq ft • {bedrooms} bed • {bathrooms} bath</p>
                    <small>Exchange Rate: 1 USD = 83.50 INR</small>
                </div>
                """, unsafe_allow_html=True)
            else:  # Both currencies
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Property Valuation</h2>
                    <h1>${predicted_price:,.0f} USD</h1>
                    <h2>₹{predicted_price_inr:,.0f} INR</h2>
                    <p>For {square_feet:,} sq ft • {bedrooms} bed • {bathrooms} bath</p>
                    <small>Exchange Rate: 1 USD = 83.50 INR</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Price breakdown
            st.markdown('<h3 class="sub-header">💰 Price Breakdown</h3>', unsafe_allow_html=True)
            
            base_price = model.intercept_
            sqft_contribution = model.coef_[0] * square_feet
            bedroom_contribution = model.coef_[1] * bedrooms
            bathroom_contribution = model.coef_[2] * bathrooms
            
            # Create breakdown based on currency selection
            if currency == "INR (₹)":
                breakdown_values = [
                    usd_to_inr(base_price),
                    usd_to_inr(sqft_contribution),
                    usd_to_inr(bedroom_contribution),
                    usd_to_inr(bathroom_contribution),
                    usd_to_inr(predicted_price)
                ]
                breakdown_descriptions = [
                    'Starting price',
                    f'{square_feet:,} sq ft × ₹{usd_to_inr(model.coef_[0]):.0f}',
                    f'{bedrooms} × ₹{usd_to_inr(model.coef_[1]):,.0f}',
                    f'{bathrooms} × ₹{usd_to_inr(model.coef_[2]):,.0f}',
                    'Final predicted price'
                ]
                text_values = [f'₹{val:,.0f}' for val in breakdown_values]
                y_title = "Price Contribution (₹)"
                chart_title = "Price Component Breakdown (INR)"
            elif currency == "Both":
                # Show INR values but with dual currency text
                breakdown_values = [
                    usd_to_inr(base_price),
                    usd_to_inr(sqft_contribution),
                    usd_to_inr(bedroom_contribution),
                    usd_to_inr(bathroom_contribution),
                    usd_to_inr(predicted_price)
                ]
                breakdown_descriptions = [
                    'Starting price',
                    f'{square_feet:,} sq ft × ${model.coef_[0]:.0f} / ₹{usd_to_inr(model.coef_[0]):.0f}',
                    f'{bedrooms} × ${model.coef_[1]:,.0f} / ₹{usd_to_inr(model.coef_[1]):,.0f}',
                    f'{bathrooms} × ${model.coef_[2]:,.0f} / ₹{usd_to_inr(model.coef_[2]):,.0f}',
                    'Final predicted price'
                ]
                # Show both currencies in text
                usd_values = [base_price, sqft_contribution, bedroom_contribution, bathroom_contribution, predicted_price]
                text_values = [f'${usd_val:,.0f}\n₹{inr_val:,.0f}' for usd_val, inr_val in zip(usd_values, breakdown_values)]
                y_title = "Price Contribution (₹)"
                chart_title = "Price Component Breakdown (USD/INR)"
            else:  # USD only
                breakdown_values = [base_price, sqft_contribution, bedroom_contribution, bathroom_contribution, predicted_price]
                breakdown_descriptions = [
                    'Starting price',
                    f'{square_feet:,} sq ft × ${model.coef_[0]:.2f}',
                    f'{bedrooms} × ${model.coef_[1]:,.0f}',
                    f'{bathrooms} × ${model.coef_[2]:,.0f}',
                    'Final predicted price'
                ]
                text_values = [f'${val:,.0f}' for val in breakdown_values]
                y_title = "Price Contribution ($)"
                chart_title = "Price Component Breakdown (USD)"
            
            breakdown_df = pd.DataFrame({
                'Component': ['Base Price', 'Square Feet', 'Bedrooms', 'Bathrooms', 'Total'],
                'Value': breakdown_values,
                'Description': breakdown_descriptions
            })
            
            # Color coding for the breakdown
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'gold']
            
            fig_breakdown = go.Figure(data=[
                go.Bar(
                    x=breakdown_df['Component'],
                    y=breakdown_df['Value'],
                    marker_color=colors,
                    text=text_values,
                    textposition='auto'
                )
            ])
            
            fig_breakdown.update_layout(
                title=chart_title,
                xaxis_title="Components",
                yaxis_title=y_title,
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        else:
            st.markdown("""
            <div class="info-box">
                <h3>👋 Welcome to PropertyPredict Pro!</h3>
                <p>Enter your property details in the sidebar and click "Get Property Valuation" to see the estimated value.</p>
                <p>Our AI model analyzes thousands of real estate transactions to provide accurate property valuations!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Model performance metrics
        st.markdown('<h3 class="sub-header">📊 Model Performance</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>R² Score</h4>
            <h2>{metrics['test_r2']:.3f}</h2>
            <p>Variance Explained</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display RMSE based on currency selection
        if currency == "INR (₹)":
            rmse_display = usd_to_inr(metrics['test_rmse'])
            rmse_text = f"₹{rmse_display:,.0f}"
        elif currency == "USD ($)":
            rmse_text = f"${metrics['test_rmse']:,.0f}"
        else:  # Both
            rmse_inr = usd_to_inr(metrics['test_rmse'])
            rmse_text = f"${metrics['test_rmse']:,.0f} / ₹{rmse_inr:,.0f}"
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>RMSE</h4>
            <h2>{rmse_text}</h2>
            <p>Prediction Error</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>Training Samples</h4>
            <h2>{len(df):,}</h2>
            <p>Houses Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations section
    st.markdown('<h2 class="sub-header">📈 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Feature Importance", "📊 Model Accuracy", "💹 Price Distribution", "🔍 Data Insights"])
    
    # Determine currency for charts - if "Both" is selected, show INR charts but with dual labels
    if currency == "INR (₹)":
        chart_currency = "INR"
    elif currency == "USD ($)":
        chart_currency = "USD"
    else:  # Both - show INR charts since they're larger numbers and more impressive
        chart_currency = "INR"
    
    with tab1:
        st.plotly_chart(create_feature_importance_chart(model, chart_currency), use_container_width=True)
        
        # Update insights based on currency
        if currency == "INR (₹)" or currency == "Both":
            sqft_value_usd = model.coef_[0]
            sqft_value_inr = usd_to_inr(sqft_value_usd)
            if currency == "Both":
                value_text = f"**${sqft_value_usd:.0f} USD / ₹{sqft_value_inr:.0f} INR**"
            else:
                value_text = f"**₹{sqft_value_inr:.0f}**"
        else:  # USD only
            sqft_value = model.coef_[0]
            value_text = f"**${sqft_value:.0f}**"
            
        st.markdown(f"""
        **Key Insights:**
        - **Square Feet** has the strongest impact on property value
        - Each additional square foot adds approximately {value_text} to the property value
        - Bedrooms and bathrooms provide additional value but with diminishing returns
        """)
    
    with tab2:
        st.plotly_chart(create_prediction_vs_actual_chart(metrics['y_test'], metrics['y_pred_test'], chart_currency), use_container_width=True)
        
        # Update performance text based on currency
        if currency == "INR (₹)":
            rmse_text = f"₹{usd_to_inr(metrics['test_rmse']):,.0f}"
        elif currency == "USD ($)":
            rmse_text = f"${metrics['test_rmse']:,.0f}"
        else:  # Both
            rmse_inr = usd_to_inr(metrics['test_rmse'])
            rmse_text = f"${metrics['test_rmse']:,.0f} / ₹{rmse_inr:,.0f}"
            
        st.markdown(f"""
        **Model Performance:**
        - **R² Score: {metrics['test_r2']:.3f}** - The model explains {metrics['test_r2']*100:.1f}% of property value variation
        - **RMSE: {rmse_text}** - Average prediction error
        - Points closer to the red line indicate more accurate predictions
        """)
    
    with tab3:
        st.plotly_chart(create_price_distribution_chart(df, chart_currency), use_container_width=True)
        
        # Update price statistics based on currency
        col1, col2, col3 = st.columns(3)
        
        if currency == "INR (₹)":
            with col1:
                avg_price_inr = usd_to_inr(df['price'].mean())
                st.metric("Average Price", f"₹{avg_price_inr:,.0f}")
            with col2:
                median_price_inr = usd_to_inr(df['price'].median())
                st.metric("Median Price", f"₹{median_price_inr:,.0f}")
            with col3:
                price_range_inr = usd_to_inr(df['price'].max() - df['price'].min())
                st.metric("Price Range", f"₹{price_range_inr:,.0f}")
        elif currency == "USD ($)":
            with col1:
                st.metric("Average Price", f"${df['price'].mean():,.0f}")
            with col2:
                st.metric("Median Price", f"${df['price'].median():,.0f}")
            with col3:
                st.metric("Price Range", f"${df['price'].max() - df['price'].min():,.0f}")
        else:  # Both currencies
            with col1:
                avg_price_inr = usd_to_inr(df['price'].mean())
                st.metric("Average Price (USD)", f"${df['price'].mean():,.0f}")
                st.metric("Average Price (INR)", f"₹{avg_price_inr:,.0f}")
            with col2:
                median_price_inr = usd_to_inr(df['price'].median())
                st.metric("Median Price (USD)", f"${df['price'].median():,.0f}")
                st.metric("Median Price (INR)", f"₹{median_price_inr:,.0f}")
            with col3:
                price_range = df['price'].max() - df['price'].min()
                price_range_inr = usd_to_inr(price_range)
                st.metric("Price Range (USD)", f"${price_range:,.0f}")
                st.metric("Price Range (INR)", f"₹{price_range_inr:,.0f}")
    
    with tab4:
        # Correlation matrix
        corr_matrix = df[['square_feet', 'bedrooms', 'bathrooms', 'price']].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Summary statistics
        st.markdown("**Dataset Summary:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏡 PropertyPredict Pro | Built with ❤️ using AI & Machine Learning</p>
        <p>Data Source: Kaggle House Prices Dataset | Model: Linear Regression</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
