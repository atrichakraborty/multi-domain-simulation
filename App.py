"""
Advanced Multi-Domain Simulation Dashboard
Quantitative Financial Engineering & Formula One Strategy Analytics
Built with Streamlit

Interactive frontend for:
- Heston Stochastic Volatility Asset Projections
- Portfolio Optimization & Efficient Frontier Analysis
- F1 Race Strategy Simulations (1-Stop vs 2-Stop)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import yfinance as yf
from engine import (
    QuantFinanceEngine, F1StrategyEngine, HestonParams, SimulationFactory
)
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Multi-Domain Simulation Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .kpi-container {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.title("⚙️ Simulation Dashboard")
domain = st.sidebar.radio(
    "Select Domain:",
    ["📈 Quantitative Finance", "🏎️ Formula One Strategy"],
    index=0
)

# ==================== DOMAIN 1: QUANTITATIVE FINANCE ====================
if domain == "📈 Quantitative Finance":
    st.title("📈 Quantitative Financial Engineering")
    st.markdown("**Advanced Stochastic Volatility Modeling & Portfolio Optimization**")
    
    # Sidebar parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Financial Simulation Parameters")
    
    tickers_input = st.sidebar.text_input(
        "Stock Tickers (comma-separated)",
        value="AAPL,MSFT,GOOGL,NVDA,TSLA",
        help="E.g., AAPL,MSFT,GOOGL"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    sim_runs = st.sidebar.slider(
        "Monte Carlo Paths",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500
    )
    
    time_horizon = st.sidebar.slider(
        "Time Horizon (Years)",
        min_value=0.25,
        max_value=3.0,
        value=1.0,
        step=0.25
    )
    
    trading_days = int(time_horizon * 252)
    
    # Heston model parameters
    st.sidebar.markdown("**Heston Model Parameters**")
    kappa = st.sidebar.slider(
        "Mean Reversion Speed (κ)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )
    theta = st.sidebar.slider(
        "Long-term Variance (θ)",
        min_value=0.01,
        max_value=0.2,
        value=0.04,
        step=0.01
    )
    sigma = st.sidebar.slider(
        "Volatility of Volatility (σ)",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.05
    )
    
    # Portfolio parameters
    n_portfolios = st.sidebar.slider(
        "Random Portfolios to Generate",
        min_value=200,
        max_value=1000,
        value=500,
        step=100
    )
    
    # Fetch data
    try:
        @st.cache_data(ttl=3600)
        def fetch_stock_data(tickers, period="1y"):
            data = yf.download(tickers, period=period, progress=False)
            return data['Close']
        
        price_data = fetch_stock_data(tickers)
        latest_prices = price_data.iloc[-1].values
        
        st.success(f"✅ Loaded {len(tickers)} tickers successfully")
        
        # Main simulation
        if st.sidebar.button("🚀 Run Simulation", key="fin_sim"):
            with st.spinner("Running Heston simulations and portfolio optimization..."):
                
                # Initialize engine
                engine = SimulationFactory.get_finance_engine(risk_free_rate=0.05)
                
                # Heston parameters
                heston_params = HestonParams(
                    kappa=kappa,
                    theta=theta,
                    sigma=sigma,
                    rho=-0.7,
                    v0=theta
                )
                
                # Run simulations
                price_paths, variance_paths = engine.heston_monte_carlo(
                    S0=latest_prices,
                    T=time_horizon,
                    steps=trading_days,
                    paths=sim_runs,
                    params=heston_params,
                    seed=42
                )
                
                # Efficient frontier
                vol, ret, sharpe, es = engine.efficient_frontier(
                    price_paths,
                    n_portfolios=n_portfolios
                )
                
                # Find optimal portfolio
                optimal_idx = np.argmax(sharpe)
                max_sharpe = sharpe[optimal_idx]
                
                # Create tabs for results
                tab1, tab2, tab3 = st.tabs(
                    ["Price Paths", "Efficient Frontier", "Risk Metrics"]
                )
                
                # TAB 1: Price Paths
                with tab1:
                    st.subheader("Heston Stochastic Volatility - Price Paths")
                    
                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Plot price paths for first asset
                    ax = axes[0]
                    for i in range(min(100, sim_runs)):
                        ax.plot(price_paths[:, i, 0], alpha=0.1, color='blue')
                    
                    # Mean path
                    mean_path = np.mean(price_paths[:, :, 0], axis=1)
                    ax.plot(mean_path, color='red', linewidth=3, label='Mean Path', alpha=0.8)
                    ax.set_xlabel('Trading Days', fontsize=11)
                    ax.set_ylabel('Price ($)', fontsize=11)
                    ax.set_title(f'{tickers[0]} - {sim_runs} Monte Carlo Paths', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    # Volatility surface
                    ax = axes[1]
                    mean_vol = np.mean(np.sqrt(variance_paths[:, :, 0]), axis=1) * np.sqrt(252) * 100
                    ax.fill_between(range(len(mean_vol)), mean_vol - np.std(np.sqrt(variance_paths[:, :, 0]), axis=1) * np.sqrt(252) * 100,
                                   mean_vol + np.std(np.sqrt(variance_paths[:, :, 0]), axis=1) * np.sqrt(252) * 100,
                                   alpha=0.3, color='green', label='±1 Std Dev')
                    ax.plot(mean_vol, color='darkgreen', linewidth=2, label='Mean Volatility')
                    ax.set_xlabel('Trading Days', fontsize=11)
                    ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
                    ax.set_title(f'{tickers[0]} - Dynamic Volatility Evolution', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                # TAB 2: Efficient Frontier
                with tab2:
                    st.subheader("Modern Portfolio Theory - Efficient Frontier")
                    
                    fig, ax = plt.subplots(figsize=(12, 7))
                    
                    # Scatter plot colored by Expected Shortfall
                    scatter = ax.scatter(
                        vol * 100,
                        ret * 100,
                        c=es * 100,
                        cmap='RdYlGn_r',
                        s=100,
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    
                    # Highlight optimal portfolio
                    ax.scatter(
                        vol[optimal_idx] * 100,
                        ret[optimal_idx] * 100,
                        color='gold',
                        s=500,
                        marker='*',
                        edgecolors='black',
                        linewidth=2,
                        label=f'Max Sharpe: {max_sharpe:.3f}',
                        zorder=5
                    )
                    
                    ax.set_xlabel('Volatility (% p.a.)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Expected Return (% p.a.)', fontsize=12, fontweight='bold')
                    ax.set_title('Efficient Frontier - Random Portfolio Sampling', fontsize=13, fontweight='bold')
                    ax.grid(alpha=0.3, linestyle='--')
                    ax.legend(fontsize=11)
                    
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Expected Shortfall (% p.a.)', fontsize=11, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                # TAB 3: Risk Metrics
                with tab3:
                    st.subheader("Tail-Risk Diagnostics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Max Sharpe Ratio",
                            f"{max_sharpe:.4f}",
                            delta=None,
                            delta_color="inverse"
                        )
                    
                    with col2:
                        avg_es = np.mean(es) * 100
                        st.metric(
                            "Avg Expected Shortfall (95%)",
                            f"{avg_es:.3f}%",
                            delta=None,
                            delta_color="off"
                        )
                    
                    with col3:
                        max_kurtosis = np.max(es) * 100
                        st.metric(
                            "Worst Case ES Portfolio",
                            f"{max_kurtosis:.3f}%",
                            delta=None,
                            delta_color="off"
                        )
                    
                    # Distribution analysis
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # Sharpe ratio distribution
                    ax = axes[0, 0]
                    ax.hist(sharpe, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
                    ax.axvline(max_sharpe, color='red', linestyle='--', linewidth=2, label=f'Max: {max_sharpe:.3f}')
                    ax.set_xlabel('Sharpe Ratio', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.set_title('Distribution of Sharpe Ratios', fontsize=11, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    # Expected Shortfall distribution
                    ax = axes[0, 1]
                    ax.hist(es * 100, bins=50, color='coral', edgecolor='black', alpha=0.7)
                    ax.axvline(np.mean(es) * 100, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(es)*100:.2f}%')
                    ax.set_xlabel('Expected Shortfall (95%) - %', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.set_title('Distribution of ES Values', fontsize=11, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    # Risk-Return scatter
                    ax = axes[1, 0]
                    ax.scatter(vol * 100, ret * 100, c=sharpe, cmap='viridis', s=80, alpha=0.6)
                    ax.scatter(vol[optimal_idx] * 100, ret[optimal_idx] * 100, color='gold', s=300, marker='*', edgecolors='black', linewidth=2)
                    ax.set_xlabel('Volatility (%)', fontsize=10)
                    ax.set_ylabel('Return (%)', fontsize=10)
                    ax.set_title('Risk-Return Trade-off', fontsize=11, fontweight='bold')
                    ax.grid(alpha=0.3)
                    
                    # Portfolio volatility vs ES
                    ax = axes[1, 1]
                    ax.scatter(vol * 100, es * 100, c=sharpe, cmap='coolwarm', s=80, alpha=0.6)
                    ax.set_xlabel('Portfolio Volatility (%)', fontsize=10)
                    ax.set_ylabel('Expected Shortfall (95%) - %', fontsize=10)
                    ax.set_title('Volatility vs Tail Risk', fontsize=11, fontweight='bold')
                    ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    
                    # Summary statistics table
                    st.markdown("**Portfolio Metrics Summary**")
                    summary_df = pd.DataFrame({
                        'Metric': ['Mean Sharpe Ratio', 'Median Sharpe Ratio', 'Std Dev (Sharpe)', 
                                  'Max Expected Shortfall (%)', 'Min Expected Shortfall (%)',
                                  'Mean Portfolio Volatility (%)'],
                        'Value': [
                            f"{np.mean(sharpe):.4f}",
                            f"{np.median(sharpe):.4f}",
                            f"{np.std(sharpe):.4f}",
                            f"{np.max(es) * 100:.3f}",
                            f"{np.min(es) * 100:.3f}",
                            f"{np.mean(vol) * 100:.3f}"
                        ]
                    })
                    st.dataframe(summary_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please check ticker symbols and try again.")


# ==================== DOMAIN 2: FORMULA ONE STRATEGY ====================
elif domain == "🏎️ Formula One Strategy":
    st.title("🏎️ Formula One Strategy Analytics")
    st.markdown("**Advanced Race Logistics Simulation: 1-Stop vs 2-Stop Strategy Comparison**")
    
    # Sidebar parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Race Simulation Parameters")
    
    race_laps = st.sidebar.slider(
        "Total Race Laps",
        min_value=20,
        max_value=78,
        value=52,
        step=2
    )
    
    base_lap_time = st.sidebar.slider(
        "Base Lap Time (seconds)",
        min_value=70.0,
        max_value=120.0,
        value=90.0,
        step=1.0
    )
    
    tire_wear_rate = st.sidebar.slider(
        "Tire Wear Rate (sec/lap)",
        min_value=0.05,
        max_value=0.5,
        value=0.15,
        step=0.05
    )
    
    fuel_burnoff = st.sidebar.slider(
        "Fuel Burn Effect (sec/lap)",
        min_value=-0.15,
        max_value=0.0,
        value=-0.06,
        step=0.01
    )
    
    pit_loss = st.sidebar.slider(
        "Pit Stop Loss Time (seconds)",
        min_value=15.0,
        max_value=40.0,
        value=25.0,
        step=1.0
    )
    
    sc_probability = st.sidebar.slider(
        "Safety Car Probability",
        min_value=0.0,
        max_value=0.8,
        value=0.3,
        step=0.1
    )
    
    sim_runs = st.sidebar.slider(
        "Simulation Runs",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500
    )
    
    # Run simulation
    if st.sidebar.button("🚀 Run Race Simulation", key="f1_sim"):
        with st.spinner("Simulating F1 race strategies across thousands of scenarios..."):
            
            # Initialize F1 engine
            f1_engine = SimulationFactory.get_f1_engine()
            
            # Compare strategies
            results = f1_engine.compare_strategies(
                n_laps=race_laps,
                base_lap_time=base_lap_time,
                tire_wear_rate=tire_wear_rate,
                fuel_effect=fuel_burnoff,
                pit_loss=pit_loss,
                sc_probability=sc_probability,
                n_simulations=sim_runs,
                seed=42
            )
            
            times_1stop = results['1-stop']
            times_2stop = results['2-stop']
            
            # Compute statistics
            mean_1stop = np.mean(times_1stop)
            mean_2stop = np.mean(times_2stop)
            std_1stop = np.std(times_1stop)
            std_2stop = np.std(times_2stop)
            
            time_delta = mean_2stop - mean_1stop
            better_strategy = "1-Stop" if time_delta > 0 else "2-Stop"
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(
                ["Race Time Distributions", "Strategy Comparison", "Detailed Analysis"]
            )
            
            # TAB 1: Distributions
            with tab1:
                st.subheader("Race Finish Time Distributions")
                
                fig, ax = plt.subplots(figsize=(14, 7))
                
                # Histograms
                ax.hist(times_1stop / 60, bins=50, alpha=0.6, label='1-Stop Strategy',
                       color='steelblue', edgecolor='black')
                ax.hist(times_2stop / 60, bins=50, alpha=0.6, label='2-Stop Strategy',
                       color='coral', edgecolor='black')
                
                # Mean lines
                ax.axvline(mean_1stop / 60, color='darkblue', linestyle='--', linewidth=2.5,
                          label=f'1-Stop Mean: {mean_1stop/60:.2f} min')
                ax.axvline(mean_2stop / 60, color='darkred', linestyle='--', linewidth=2.5,
                          label=f'2-Stop Mean: {mean_2stop/60:.2f} min')
                
                ax.set_xlabel('Race Time (minutes)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title(f'Race Finish Time Distributions ({sim_runs:,} Simulations)', 
                           fontsize=13, fontweight='bold')
                ax.legend(fontsize=11, loc='upper right')
                ax.grid(alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            # TAB 2: Strategy Comparison
            with tab2:
                st.subheader("Strategy Performance Metrics")
                
                # KPI Cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Optimal Strategy",
                        better_strategy,
                        delta=f"{abs(time_delta):.2f}s advantage" if abs(time_delta) > 1 else "Marginal difference"
                    )
                
                with col2:
                    st.metric(
                        "1-Stop Avg Time",
                        f"{mean_1stop/60:.2f} min",
                        delta=f"{std_1stop/60:.3f} min std dev"
                    )
                
                with col3:
                    st.metric(
                        "2-Stop Avg Time",
                        f"{mean_2stop/60:.2f} min",
                        delta=f"{std_2stop/60:.3f} min std dev"
                    )
                
                # Comparison statistics
                st.markdown("**Statistical Summary**")
                comp_df = pd.DataFrame({
                    'Metric': [
                        'Mean Race Time (min)',
                        'Std Deviation (min)',
                        '5th Percentile (min)',
                        'Median (min)',
                        '95th Percentile (min)',
                        'Best Case (min)',
                        'Worst Case (min)'
                    ],
                    '1-Stop': [
                        f"{mean_1stop/60:.3f}",
                        f"{std_1stop/60:.3f}",
                        f"{np.percentile(times_1stop, 5)/60:.3f}",
                        f"{np.median(times_1stop)/60:.3f}",
                        f"{np.percentile(times_1stop, 95)/60:.3f}",
                        f"{np.min(times_1stop)/60:.3f}",
                        f"{np.max(times_1stop)/60:.3f}"
                    ],
                    '2-Stop': [
                        f"{mean_2stop/60:.3f}",
                        f"{std_2stop/60:.3f}",
                        f"{np.percentile(times_2stop, 5)/60:.3f}",
                        f"{np.median(times_2stop)/60:.3f}",
                        f"{np.percentile(times_2stop, 95)/60:.3f}",
                        f"{np.min(times_2stop)/60:.3f}",
                        f"{np.max(times_2stop)/60:.3f}"
                    ]
                })
                st.dataframe(comp_df, use_container_width=True)
                
                # Strategy recommendation
                st.markdown("---")
                recommendation_text = f"""
                ### 📊 Strategy Recommendation
                
                **Optimal Strategy:** **{better_strategy}**
                
                **Time Advantage:** {abs(time_delta):.2f} seconds ({abs(time_delta/mean_2stop)*100:.2f}% improvement)
                
                **Confidence:** Based on {sim_runs:,} Monte Carlo simulations across varying race conditions including:
                - Tire degradation with exponential wear compounding
                - Dynamic fuel weight reduction over race distance
                - Stochastic safety car events ({sc_probability*100:.0f}% probability)
                - Traffic delays on worn tire compounds
                
                **Race Parameters:**
                - Total Laps: {race_laps}
                - Base Lap Time: {base_lap_time:.1f}s
                - Tire Wear Rate: {tire_wear_rate:.3f}s/lap
                - Pit Stop Time: {pit_loss:.1f}s
                """
                st.markdown(recommendation_text)
            
            # TAB 3: Detailed Analysis
            with tab3:
                st.subheader("Advanced Analysis & Distributions")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Box plots
                ax = axes[0, 0]
                bp_data = [times_1stop/60, times_2stop/60]
                bp = ax.boxplot(bp_data, labels=['1-Stop', '2-Stop'], patch_artist=True,
                               boxprops=dict(facecolor='lightblue'),
                               medianprops=dict(color='red', linewidth=2))
                ax.set_ylabel('Race Time (minutes)', fontsize=10, fontweight='bold')
                ax.set_title('Race Time Distributions (Box Plot)', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
                
                # Q-Q comparison
                ax = axes[0, 1]
                sorted_1stop = np.sort(times_1stop/60)
                sorted_2stop = np.sort(times_2stop/60)
                ax.scatter(sorted_1stop, sorted_2stop, alpha=0.5, s=20)
                min_val = min(sorted_1stop.min(), sorted_2stop.min())
                max_val = max(sorted_1stop.max(), sorted_2stop.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance')
                ax.set_xlabel('1-Stop Quantiles (min)', fontsize=10, fontweight='bold')
                ax.set_ylabel('2-Stop Quantiles (min)', fontsize=10, fontweight='bold')
                ax.set_title('Q-Q Plot: Strategy Comparison', fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                
                # Cumulative distribution
                ax = axes[1, 0]
                ax.hist(times_1stop/60, bins=50, cumulative=True, density=True, alpha=0.6,
                       label='1-Stop', color='steelblue', edgecolor='black')
                ax.hist(times_2stop/60, bins=50, cumulative=True, density=True, alpha=0.6,
                       label='2-Stop', color='coral', edgecolor='black')
                ax.set_xlabel('Race Time (minutes)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
                ax.set_title('Cumulative Distribution Function', fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                
                # Probability density
                ax = axes[1, 1]
                from scipy import stats
                kde_1stop = stats.gaussian_kde(times_1stop/60)
                kde_2stop = stats.gaussian_kde(times_2stop/60)
                x_range = np.linspace(min(times_1stop.min(), times_2stop.min())/60,
                                     max(times_1stop.max(), times_2stop.max())/60, 200)
                ax.plot(x_range, kde_1stop(x_range), linewidth=2.5, label='1-Stop KDE', color='steelblue')
                ax.plot(x_range, kde_2stop(x_range), linewidth=2.5, label='2-Stop KDE', color='coral')
                ax.fill_between(x_range, kde_1stop(x_range), alpha=0.3, color='steelblue')
                ax.fill_between(x_range, kde_2stop(x_range), alpha=0.3, color='coral')
                ax.set_xlabel('Race Time (minutes)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
                ax.set_title('Kernel Density Estimation', fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Advanced Multi-Domain Simulation Platform</strong></p>
        <p>Built with Streamlit | Powered by NumPy, Pandas & Matplotlib</p>
        <p>Featuring Heston Stochastic Volatility & F1 Strategy Engine</p>
    </div>
""", unsafe_allow_html=True)
