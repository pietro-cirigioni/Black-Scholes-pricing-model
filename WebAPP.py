import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

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
        <p>Welcome to my Black-Scholes Pricing Model app! You can set the parameters on the left and watch the option prices and heatmaps change dynamically.</p>
        <p>I'm a quantitative finance student passionate about Investment Strategies, Research and Portfolio Theory.</p>
        <p>Connect with me on <a href='https://www.linkedin.com/in/pietro-cirigioni/' target='_blank'>LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

# Title and header
st.title('Black-Scholes Pricing Model')

# Sidebar Inputs for Black-Scholes parameters
st.sidebar.header("Model Parameters")

# Inputs for Stock Price, Strike Price, Time to Maturity, Risk-free Rate, and Volatility
S = st.sidebar.number_input('Stock Price (S)', value=100.00)
K = st.sidebar.number_input('Strike Price (K)', value=100.00)
T = st.sidebar.number_input('Time to Maturity (T in years)', value=1.00)
r = st.sidebar.number_input('Risk-free Rate (r)', value=0.05)
selected_volatility = st.sidebar.number_input('Volatility (σ)', value=0.25, min_value=0.01, max_value=1.0)

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

# Determine dynamic range for spot price slider
min_spot = float(max(0, S - 20))  # Ensure min_spot is float
max_spot = float(max(S + 20, 0))  # Ensure max_spot is float

# Add Sliders for Min/Max Spot Price
selected_min_spot, selected_max_spot = st.sidebar.slider(
    'Spot Price Range (S)', min_spot, max_spot, (min_spot, max_spot), step=1.0
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
S_values = np.linspace(selected_min_spot, selected_max_spot, 10)  # Spot price values
sigma_values = np.linspace(min_volatility, max_volatility, 10)  # Volatility values

# Prepare arrays for prices
prices_call = np.zeros((len(S_values), len(sigma_values)))
prices_put = np.zeros((len(S_values), len(sigma_values)))

# Calculate call and put prices
for i, S_val in enumerate(S_values):
    for j, sigma_val in enumerate(sigma_values):
        prices_call[i, j] = Black_Scholes_call(S_val, K, T, r, sigma_val)
        prices_put[i, j] = Black_Scholes_put(S_val, K, T, r, sigma_val)

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
cax_call = ax_call.imshow(prices_call, aspect='auto', cmap='plasma_r', origin='lower',
                          extent=[selected_min_spot, selected_max_spot, min_volatility, max_volatility])
ax_call.set_title('Call Prices Heatmap')
ax_call.set_xlabel('Spot Price (S)')
ax_call.set_ylabel('Volatility (σ)')
fig_call.colorbar(cax_call, label='Call Price')

# Flip the data along the spot price axis (axis=1) without flipping the y-axis (volatility)
prices_put_flipped_x = np.flip(prices_put, axis=1)

# Plotting Put Prices Heatmap
fig_put, ax_put = plt.subplots()
cax_put = ax_put.imshow(prices_put_flipped_x, aspect='auto', cmap='viridis_r', origin='upper',
                        extent=[selected_min_spot, selected_max_spot, min_volatility, max_volatility])
ax_put.set_title('Put Prices Heatmap')
ax_put.set_xlabel('Spot Price (S)')
ax_put.set_ylabel('Volatility (σ)')
fig_put.colorbar(cax_put, label='Put Price')

# Display both heatmaps side by side
col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig_call)
with col4:
    st.pyplot(fig_put)

# Generate 3D surface for Call prices
fig_call_3d = plt.figure(figsize=(12, 8))
ax_call_3d = fig_call_3d.add_subplot(111, projection='3d')

# Prepare meshgrid and calculate Call prices
X, Y = np.meshgrid(S_values, sigma_values)
Z_call = np.array([[Black_Scholes_call(S_val, K, T, r, sigma_val) 
                   for sigma_val in sigma_values] for S_val in S_values])

# Plot surface
ax_call_3d.plot_surface(X, Y, Z_call.T, cmap='winter_r', edgecolor='none')

# Set axis labels and titles
ax_call_3d.set_title('Call Prices 3D Surface')
ax_call_3d.set_xlabel('Spot Price (S)')  # Spot Price on X-axis
ax_call_3d.set_ylabel('Volatility (σ)')  # Volatility on Y-axis
ax_call_3d.set_zlabel('Call Price')

# Adjust the viewing angle
ax_call_3d.view_init(elev=30, azim=60)

# Generate 3D surface for Put prices
fig_put_3d = plt.figure(figsize=(12, 8))
ax_put_3d = fig_put_3d.add_subplot(111, projection='3d')

# Prepare meshgrid and calculate Put prices
X, Y = np.meshgrid(S_values, sigma_values)
Z_put = np.array([[Black_Scholes_put(S_val, K, T, r, sigma_val) 
                   for sigma_val in sigma_values] for S_val in S_values])

# Plot surface
ax_put_3d.plot_surface(X, Y, Z_put.T, cmap='winter_r', edgecolor='none')

# Set axis labels and titles
ax_put_3d.set_title('Put Prices 3D Surface')
ax_put_3d.set_xlabel('Spot Price (S)')  # Spot Price on X-axis (same as Call plot)
ax_put_3d.set_ylabel('Volatility (σ)')  # Volatility on Y-axis (same as Call plot)
ax_put_3d.set_zlabel('Put Price')

# Adjust the viewing angle
ax_put_3d.view_init(elev=30, azim=60)

# Show the 3D surface plot
#st.pyplot(fig_put_3d)
# Display both 3D plots side by side
col5, col6 = st.columns(2)
with col5:
    st.pyplot(fig_call_3d)
with col6:
    st.pyplot(fig_put_3d)
