import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set page config
st.set_page_config(page_title="Black-Scholes Pricing Model", layout="wide")

# Custom CSS for layout and theming
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #F0F0F0;
    }
    .sidebar .sidebar-content {
        background-color: #242526;
        color: #F0F0F0;
    }
    h1, h2, h3 {
        color: #F0F0F0;
    }
    .metric {
        font-size: 1.5em;
        padding: 1em;
    }
    </style>
    <div style='text-align: center;'>
        <h1>Pietro Cirigioni</h1>
        <p>Welcome to my Black-Scholes Pricing Model app! You can set the parameters on the left and watch the option prices and heatmaps change dinamically.</p>
        <p>I'm a quantitative finance student passionate about Investment Strategies, Research and Portfolio Theory.</p>
        <p>Connect with me on <a href='https://www.https://www.linkedin.com/in/pietro-cirigioni/' target='_blank'>LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

# Title and header
st.title('Black-Scholes Pricing Model')

# Sidebar Inputs for Black-Scholes parameters
st.sidebar.header("Black-Scholes Model Parameters")

# Inputs for Stock Price, Strike Price, Time to Maturity, Risk-free Rate, and Volatility
S = st.sidebar.number_input('Stock Price (S)', value=100.00)
K = st.sidebar.number_input('Strike Price (K)', value=100.00)
T = st.sidebar.number_input('Time to Maturity (T in years)', value=1.00)
r = st.sidebar.number_input('Risk-free Rate (r)', value=0.05)
selected_volatility = st.sidebar.number_input('Volatility for Option Prices (σ)', value=0.25, min_value=0.01, max_value=1.0)

# Add a horizontal line
st.sidebar.markdown("---")

# Determine the maximum value for the volatility slider
if selected_volatility >= 0.67:
    max_volatility = selected_volatility * 1.5
else:
    max_volatility = 1.0  # or whatever you want as a default max

# Add Sliders for Min/Max Volatility
min_volatility, max_volatility = st.sidebar.slider(
    'Volatility Range (σ)', 0.0, max_volatility, (0.01, 0.5), step=0.01
)

# Determine dynamic range for strike price slider
min_strike = float(max(0, K - 50))  # Ensure min_strike is float
max_strike = float(max(K + 50, 200))  # Ensure max_strike is float

# Add Sliders for Min/Max Strike Price
selected_min_strike, selected_max_strike = st.sidebar.slider(
    'Strike Price Range (K)', min_strike, max_strike, (min_strike, max_strike), step=1.0
)

# Determine dynamic range for stock price slider
min_stock_price = float(max(0, S - 50))  # Ensure min_stock_price is float
max_stock_price = float(max(S + 50, 200))  # Ensure max_stock_price is float

# Add Sliders for Min/Max Stock Price
selected_min_stock_price, selected_max_stock_price = st.sidebar.slider(
    'Stock Price Range (S)', min_stock_price, max_stock_price, (min_stock_price, max_stock_price), step=1.0
)

# Black-Scholes formula for Call and Put prices
def Black_Scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def Black_Scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Define the range for strike prices and volatilities
K_values = np.linspace(selected_min_strike, selected_max_strike, 10)  # Strike price values
sigma_values = np.linspace(min_volatility, max_volatility, 10)  # Volatility values

# Prepare arrays for prices
prices_call = np.zeros((len(K_values), len(sigma_values)))
prices_put = np.zeros((len(K_values), len(sigma_values)))

# Calculate call and put prices
for i, K_val in enumerate(K_values):
    for j, sigma_val in enumerate(sigma_values):
        prices_call[i, j] = Black_Scholes_call(S, K_val, T, r, sigma_val)
        prices_put[i, j] = Black_Scholes_put(S, K_val, T, r, sigma_val)

# Compute Call and Put Prices with the user-specified Strike Price and Volatility
call_price = Black_Scholes_call(S, K, T, r, selected_volatility)
put_price = Black_Scholes_put(S, K, T, r, selected_volatility)

# Display Call and Put Prices
st.markdown("---")
col1, col2 = st.columns(2)
col1.metric(label="CALL Value", value=f"${call_price:.2f}", delta="")
col2.metric(label="PUT Value", value=f"${put_price:.2f}", delta="")

st.markdown("---")
st.subheader('Options Price - Interactive Heatmap')

# Plotting Call Prices Heatmap
fig_call, ax_call = plt.subplots()
cax_call = ax_call.imshow(prices_call, aspect='auto', cmap='plasma', origin='lower',
                          extent=[selected_min_strike, selected_max_strike, min_volatility, max_volatility])  # Adjusted extent
ax_call.set_title('Call Prices Heatmap')
ax_call.set_xlabel('Strike Price (K)')  # Adjusted label
ax_call.set_ylabel('Volatility (σ)')  # Adjusted label
fig_call.colorbar(cax_call, label='Call Price')

# Plotting Put Prices Heatmap
fig_put, ax_put = plt.subplots()
cax_put = ax_put.imshow(prices_put, aspect='auto', cmap='viridis', origin='lower',
                        extent=[selected_min_strike, selected_max_strike, min_volatility, max_volatility])  # Adjusted extent
ax_put.set_title('Put Prices Heatmap')
ax_put.set_xlabel('Strike Price (K)')  # Adjusted label
ax_put.set_ylabel('Volatility (σ)')  # Adjusted label
fig_put.colorbar(cax_put, label='Put Price')

# Display both heatmaps side by side
col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig_call)
with col4:
    st.pyplot(fig_put)
