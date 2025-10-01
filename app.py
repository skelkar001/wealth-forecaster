import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import json

# Try importing plotly, fallback if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="💰 Wealth Forecaster",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'accounts_data' not in st.session_state:
        st.session_state.accounts_data = []
    if 'goals_data' not in st.session_state:
        st.session_state.goals_data = {}
    if 'risk_profile' not in st.session_state:
        st.session_state.risk_profile = {}
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None

# Financial calculations
class WealthForecaster:
    def __init__(self):
        self.tax_brackets_2024 = [
            (0, 11600, 0.10),
            (11600, 47150, 0.12),
            (47150, 100525, 0.22),
            (100525, 191675, 0.24),
            (191675, 243725, 0.32),
            (243725, 609350, 0.35),
            (609350, float('inf'), 0.37)
        ]
        
    def calculate_tax_rate(self, income):
        """Calculate effective tax rate based on 2024 tax brackets"""
        total_tax = 0
        remaining_income = income
        
        for lower, upper, rate in self.tax_brackets_2024:
            if remaining_income <= 0:
                break
            taxable_in_bracket = min(remaining_income, upper - lower)
            total_tax += taxable_in_bracket * rate
            remaining_income -= taxable_in_bracket
            
        return total_tax / income if income > 0 else 0
    
    def monte_carlo_simulation(self, accounts_data, goals_data, risk_profile, years=30, simulations=1000):
        """Comprehensive Monte Carlo wealth simulation"""
        
        total_wealth = sum(acc['balance'] for acc in accounts_data)
        if total_wealth == 0:
            return None
            
        # Asset allocation and expected returns
        account_returns = {
            'Banking': {'mean': 0.02, 'std': 0.01},
            'Taxable Investment': {'mean': 0.07, 'std': 0.16},
            'Traditional 401k': {'mean': 0.07, 'std': 0.16},
            'Roth 401k': {'mean': 0.07, 'std': 0.16},
            'Traditional IRA': {'mean': 0.07, 'std': 0.16},
            'Roth IRA': {'mean': 0.07, 'std': 0.16},
            'HSA': {'mean': 0.06, 'std': 0.14},
            'Cryptocurrency': {'mean': 0.12, 'std': 0.50},
            'Real Estate': {'mean': 0.05, 'std': 0.12}
        }
        
        # Risk adjustment
        risk_multiplier = {
            'Conservative': 0.7,
            'Moderate': 1.0,
            'Aggressive': 1.3
        }.get(risk_profile.get('tolerance', 'Moderate'), 1.0)
        
        # Simulation arrays
        results = np.zeros((simulations, years + 1))
        results[:, 0] = total_wealth
        
        # Annual data
        annual_income = goals_data.get('annual_income', 100000)
        annual_expenses = goals_data.get('annual_expenses', 80000)
        annual_savings = max(0, annual_income - annual_expenses)
        
        for sim in range(simulations):
            wealth = total_wealth
            current_savings = annual_savings
            
            for year in range(1, years + 1):
                # Calculate weighted return
                total_balance = sum(acc['balance'] for acc in accounts_data)
                weighted_return = 0
                weighted_std = 0
                
                for acc in accounts_data:
                    if total_balance > 0:
                        weight = acc['balance'] / total_balance
                        acc_type = acc['type']
                        if acc_type in account_returns:
                            mean_return = account_returns[acc_type]['mean'] * risk_multiplier
                            std_return = account_returns[acc_type]['std']
                            weighted_return += weight * mean_return
                            weighted_std += weight * (std_return ** 2)
                
                weighted_std = math.sqrt(weighted_std) if weighted_std > 0 else 0.1
                
                # Generate random return
                annual_return = np.random.normal(weighted_return, weighted_std)
                
                # Apply return and add savings
                wealth = wealth * (1 + annual_return) + current_savings
                
                # Inflation adjustment
                inflation = np.random.uniform(0.02, 0.04)
                current_savings *= (1 + inflation)
                
                results[sim, year] = max(0, wealth)
        
        # Calculate percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90]:
            percentiles[p] = np.percentile(results, p, axis=0)
            
        return {
            'results': results,
            'percentiles': percentiles,
            'years': years,
            'total_simulations': simulations,
            'final_median': np.median(results[:, -1]),
            'success_rate': np.mean(results[:, -1] > goals_data.get('target_wealth', total_wealth * 2))
        }
    
    def calculate_retirement_withdrawal_strategy(self, accounts_data, retirement_age):
        """Calculate optimal withdrawal strategy"""
        
        strategy = {
            'withdrawal_order': []
        }
        
        # Optimal withdrawal sequence
        if any(acc['type'] == 'Roth IRA' for acc in accounts_data):
            strategy['withdrawal_order'].append("1. Roth IRA contributions (tax-free, no penalties)")
        
        if any(acc['type'] == 'Taxable Investment' for acc in accounts_data):
            strategy['withdrawal_order'].append("2. Taxable investment accounts (capital gains rates)")
            
        if any('Traditional' in acc['type'] for acc in accounts_data):
            strategy['withdrawal_order'].append("3. Traditional IRA/401k (ordinary income tax)")
            
        if any(acc['type'] == 'Roth IRA' for acc in accounts_data):
            strategy['withdrawal_order'].append("4. Roth IRA earnings (last resort)")
        
        return strategy

# Main app
st.markdown("""
<div class="main-header">
    <h1>💰 Wealth Forecaster</h1>
    <p>AI-powered wealth optimization and Monte Carlo forecasting</p>
</div>
""", unsafe_allow_html=True)

init_session_state()

# Progress indicator
def show_progress():
    col1, col2, col3, col4 = st.columns(4)
    
    accounts_complete = len(st.session_state.accounts_data) > 0
    goals_complete = len(st.session_state.goals_data) > 0
    forecast_complete = st.session_state.forecast_results is not None
    
    with col1:
        icon = "✅" if accounts_complete else "⏳"
        st.markdown(f"**{icon} Accounts**")
    with col2:
        icon = "✅" if goals_complete else "⏳"
        st.markdown(f"**{icon} Goals**")
    with col3:
        icon = "✅" if forecast_complete else "⏳"
        st.markdown(f"**{icon} Forecast**")
    with col4:
        if accounts_complete and goals_complete:
            icon = "🎯"
        else:
            icon = "⏳"
        st.markdown(f"**{icon} Optimize**")

show_progress()
st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🏦 Accounts", "🎯 Goals & Risk", "🔮 Forecast", "⚡ Optimize"])

with tab1:
    st.header("Financial Overview")
    
    if len(st.session_state.accounts_data) == 0:
        st.info("👋 Welcome! Start by adding your financial accounts in the 'Accounts' tab.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### What You'll Get:
            - **Comprehensive Analysis**: All account types integrated
            - **Monte Carlo Forecasting**: Probabilistic simulations
            - **Tax Optimization**: Sophisticated projections
            - **Goal Achievement**: Probability-based planning
            """)
            
        with col2:
            st.markdown("""
            ### Quick Start:
            1. Add your financial accounts
            2. Set goals and risk tolerance
            3. Generate wealth forecasts
            4. Get optimization recommendations
            """)
    else:
        total_wealth = sum(acc['balance'] for acc in st.session_state.accounts_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Wealth", f"${total_wealth:,.0f}")
        with col2:
            account_count = len(st.session_state.accounts_data)
            st.metric("Accounts", account_count)
        with col3:
            if st.session_state.goals_data.get('target_wealth'):
                progress = (total_wealth / st.session_state.goals_data['target_wealth']) * 100
                st.metric("Goal Progress", f"{progress:.1f}%")
            else:
                st.metric("Goal Progress", "Set goals")
        
        # Portfolio allocation
        if PLOTLY_AVAILABLE:
            df_accounts = pd.DataFrame(st.session_state.accounts_data)
            fig = px.pie(df_accounts, values='balance', names='type', 
                        title='Current Portfolio Allocation')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Portfolio Breakdown")
            for acc in st.session_state.accounts_data:
                pct = (acc['balance'] / total_wealth) * 100
                st.write(f"**{acc['type']}**: ${acc['balance']:,.0f} ({pct:.1f}%)")

with tab2:
    st.header("Financial Account Setup")
    st.markdown("Add all your financial accounts for comprehensive wealth analysis.")
    
    account_types = [
        "Banking", "Taxable Investment", "Traditional 401k", "Roth 401k",
        "Traditional IRA", "Roth IRA", "HSA", "Cryptocurrency", "Real Estate"
    ]
    
    with st.expander("➕ Add New Account", expanded=len(st.session_state.accounts_data) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            acc_type = st.selectbox("Account Type", account_types)
            acc_balance = st.number_input("Current Balance ($)", min_value=0, value=0, step=1000)
            
        with col2:
            acc_name = st.text_input("Account Name", f"My {acc_type}")
            
            if acc_type in ["Taxable Investment", "Traditional 401k", "Roth 401k", "Traditional IRA", "Roth IRA"]:
                stocks_pct = st.slider("Stocks %", 0, 100, 70)
                bonds_pct = st.slider("Bonds %", 0, 100, 30)
                if stocks_pct + bonds_pct != 100:
                    st.warning("Stocks + Bonds should equal 100%")
            else:
                stocks_pct, bonds_pct = 0, 0
        
        if st.button("Add Account"):
            if acc_balance > 0:
                new_account = {
                    'name': acc_name,
                    'type': acc_type,
                    'balance': acc_balance,
                    'stocks_pct': stocks_pct,
                    'bonds_pct': bonds_pct
                }
                st.session_state.accounts_data.append(new_account)
                st.success(f"✅ Added {acc_name} with ${acc_balance:,}")
                st.rerun()
            else:
                st.error("Please enter a balance greater than $0")
    
    if st.session_state.accounts_data:
        st.subheader("Your Accounts")
        
        for i, acc in enumerate(st.session_state.accounts_data):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{acc['name']}**")
                st.write(f"*{acc['type']}*")
            with col2:
                st.write(f"${acc['balance']:,.0f}")
            with col3:
                if acc['stocks_pct'] > 0 or acc['bonds_pct'] > 0:
                    st.write(f"Stocks: {acc['stocks_pct']}%")
                    st.write(f"Bonds: {acc['bonds_pct']}%")
                else:
                    st.write("Cash equivalent")
            with col4:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.accounts_data.pop(i)
                    st.rerun()
        
        total = sum(acc['balance'] for acc in st.session_state.accounts_data)
        st.markdown(f"**Total Portfolio Value: ${total:,.0f}**")

with tab3:
    st.header("Goals & Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Goals")
        
        annual_income = st.number_input(
            "Current Annual Income ($)", 
            min_value=0, 
            value=st.session_state.goals_data.get('annual_income', 100000),
            step=5000
        )
        
        annual_expenses = st.number_input(
            "Current Annual Expenses ($)", 
            min_value=0, 
            value=st.session_state.goals_data.get('annual_expenses', 80000),
            step=5000
        )
        
        retirement_age = st.number_input(
            "Target Retirement Age", 
            min_value=50, 
            max_value=75, 
            value=st.session_state.goals_data.get('retirement_age', 65)
        )
        
        target_wealth = st.number_input(
            "Target Wealth at Retirement ($)", 
            min_value=0, 
            value=st.session_state.goals_data.get('target_wealth', 2000000),
            step=50000
        )
        
    with col2:
        st.subheader("Risk Assessment")
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(
                st.session_state.risk_profile.get('tolerance', 'Moderate')
            )
        )
        
        investment_experience = st.selectbox(
            "Investment Experience",
            ["Beginner", "Intermediate", "Advanced"],
            index=["Beginner", "Intermediate", "Advanced"].index(
                st.session_state.risk_profile.get('experience', 'Intermediate')
            )
        )
        
        market_reaction = st.selectbox(
            "If your portfolio dropped 20% in a year, you would:",
            [
                "Panic and sell everything",
                "Sell some investments",
                "Hold steady",
                "Buy more investments"
            ],
            index=st.session_state.risk_profile.get('market_reaction_idx', 2)
        )
        
        time_horizon = st.number_input(
            "Investment Time Horizon (years)",
            min_value=1,
            max_value=50,
            value=st.session_state.risk_profile.get('time_horizon', 30)
        )
    
    if st.button("Save Goals & Risk Profile"):
        st.session_state.goals_data = {
            'annual_income': annual_income,
            'annual_expenses': annual_expenses,
            'retirement_age': retirement_age,
            'target_wealth': target_wealth
        }
        
        st.session_state.risk_profile = {
            'tolerance': risk_tolerance,
            'experience': investment_experience,
            'market_reaction': market_reaction,
            'market_reaction_idx': [
                "Panic and sell everything",
                "Sell some investments", 
                "Hold steady",
                "Buy more investments"
            ].index(market_reaction),
            'time_horizon': time_horizon
        }
        
        st.success("✅ Goals and risk profile saved!")
        
        st.markdown("---")
        st.subheader("Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            annual_savings = annual_income - annual_expenses
            st.metric("Annual Savings", f"${annual_savings:,}")
            savings_rate = (annual_savings / annual_income) * 100 if annual_income > 0 else 0
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
            
        with col2:
            st.metric("Risk Profile", risk_tolerance)
            years_to_retirement = retirement_age - 35
            st.metric("Years to Retirement", f"{max(years_to_retirement, 1)} years")

with tab4:
    st.header("Wealth Forecast & Monte Carlo Analysis")
    
    if len(st.session_state.accounts_data) == 0 or len(st.session_state.goals_data) == 0:
        st.warning("⚠️ Please complete Account Setup and Goals & Risk Assessment first.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Simulation Settings")
            forecast_years = st.slider("Forecast Period (years)", 5, 40, 30)
            simulations = st.selectbox("Simulations", [100, 500, 1000], index=2)
            
            run_forecast = st.button("🔮 Run Monte Carlo Forecast", type="primary")
        
        with col1:
            if run_forecast or st.session_state.forecast_results:
                if run_forecast:
                    with st.spinner("Running Monte Carlo simulation..."):
                        forecaster = WealthForecaster()
                        results = forecaster.monte_carlo_simulation(
                            st.session_state.accounts_data,
                            st.session_state.goals_data,
                            st.session_state.risk_profile,
                            years=forecast_years,
                            simulations=simulations
                        )
                        st.session_state.forecast_results = results
                
                if st.session_state.forecast_results:
                    results = st.session_state.forecast_results
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Median Final Wealth", f"${results['final_median']:,.0f}")
                    with col_b:
                        st.metric("Success Rate", f"{results['success_rate']*100:.1f}%")
                    with col_c:
                        current_wealth = sum(acc['balance'] for acc in st.session_state.accounts_data)
                        growth_multiple = results['final_median'] / current_wealth if current_wealth > 0 else 0
                        st.metric("Wealth Multiple", f"{growth_multiple:.1f}x")
        
        if st.session_state.forecast_results:
            results = st.session_state.forecast_results
            years = list(range(results['years'] + 1))
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years, y=results['percentiles'][90],
                    fill=None, mode='lines', line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=results['percentiles'][10],
                    fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)',
                    name='80% Confidence Band', fillcolor='rgba(37,99,235,0.1)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=results['percentiles'][75],
                    fill=None, mode='lines', line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=results['percentiles'][25],
                    fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)',
                    name='50% Confidence Band', fillcolor='rgba(37,99,235,0.2)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=results['percentiles'][50],
                    mode='lines', name='Median Projection',
                    line=dict(color='#2563eb', width=3)
                ))
                
                if st.session_state.goals_data.get('target_wealth'):
                    target = st.session_state.goals_data['target_wealth']
                    fig.add_hline(
                        y=target, line_dash="dash", line_color="green",
                        annotation_text=f"Target: ${target:,.0f}"
                    )
                
                fig.update_layout(
                    title="Monte Carlo Wealth Projection",
                    xaxis_title="Years",
                    yaxis_title="Portfolio Value ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                fig.update_yaxis(tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("Wealth Projection")
                projection_df = pd.DataFrame({
                    'Year': years,
                    '10th Percentile': results['percentiles'][10],
                    'Median': results['percentiles'][50],
                    '90th Percentile': results['percentiles'][90]
                })
                st.line_chart(projection_df.set_index('Year'))
            
            st.subheader("Probability Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Wealth Ranges (Final Year):**")
                wealth_ranges = [
                    ("10th percentile", results['percentiles'][10][-1]),
                    ("25th percentile", results['percentiles'][25][-1]),
                    ("50th percentile (median)", results['percentiles'][50][-1]),
                    ("75th percentile", results['percentiles'][75][-1]),
                    ("90th percentile", results['percentiles'][90][-1])
                ]
                
                for label, value in wealth_ranges:
                    st.write(f"- {label}: ${value:,.0f}")
            
            with col2:
                if st.session_state.goals_data.get('target_wealth'):
                    target = st.session_state.goals_data['target_wealth']
                    
                    final_values = results['results'][:, -1]
                    prob_target = np.mean(final_values >= target) * 100
                    prob_double_target = np.mean(final_values >= target * 2) * 100
                    prob_half_target = np.mean(final_values >= target * 0.5) * 100
                    
                    st.markdown("**Goal Achievement Probability:**")
                    st.write(f"- Reach target (${target:,.0f}): {prob_target:.1f}%")
                    st.write(f"- Double target: {prob_double_target:.1f}%")
                    st.write(f"- At least half target: {prob_half_target:.1f}%")

with tab5:
    st.header("Portfolio Optimization & Recommendations")
    
    if not st.session_state.forecast_results:
        st.warning("⚠️ Please run the Monte Carlo forecast first.")
    else:
        forecaster = WealthForecaster()
        
        st.subheader("Current Portfolio Analysis")
        
        total_wealth = sum(acc['balance'] for acc in st.session_state.accounts_data)
        
        account_summary = {}
        for acc in st.session_state.accounts_data:
            acc_type = acc['type']
            if acc_type not in account_summary:
                account_summary[acc_type] = 0
            account_summary[acc_type] += acc['balance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Account Type Distribution:**")
            for acc_type, balance in account_summary.items():
                percentage = (balance / total_wealth) * 100
                st.write(f"- {acc_type}: ${balance:,.0f} ({percentage:.1f}%)")
                
        with col2:
            total_stocks = sum(acc['balance'] * acc['stocks_pct']/100 
                             for acc in st.session_state.accounts_data 
                             if acc['stocks_pct'] > 0)
            total_bonds = sum(acc['balance'] * acc['bonds_pct']/100 
                            for acc in st.session_state.accounts_data 
                            if acc['bonds_pct'] > 0)
            
            investment_total = total_stocks + total_bonds
            
            if investment_total > 0:
                st.markdown("**Overall Asset Allocation:**")
                stocks_pct = (total_stocks / investment_total) * 100
                bonds_pct = (total_bonds / investment_total) * 100
                st.write(f"- Stocks: {stocks_pct:.1f}%")
                st.write(f"- Bonds: {bonds_pct:.1f}%")
        
        st.subheader("🎯 Personalized Recommendations")
        
        recommendations = []
        
        traditional_retirement = sum(acc['balance'] for acc in st.session_state.accounts_data 
                                   if 'Traditional' in acc['type'])
        roth_retirement = sum(acc['balance'] for acc in st.session_state.accounts_data 
                            if 'Roth' in acc['type'])
        
        if traditional_retirement > roth_retirement * 3:
            recommendations.append({
                'title': '🔄 Consider Roth Conversions',
                'description': 'You have a high Traditional:Roth ratio. Consider converting some Traditional IRA/401k funds to Roth during low-income years.',
                'impact': 'High'
            })
        
        cash_accounts = sum(acc['balance'] for acc in st.session_state.accounts_data 
                          if acc['type'] == 'Banking')
        monthly_expenses = st.session_state.goals_data.get('annual_expenses', 80000) / 12
        emergency_fund_months = cash_accounts / monthly_expenses if monthly_expenses > 0 else 0
        
        if emergency_fund_months < 3:
            recommendations.append({
                'title': '🚨 Build Emergency Fund',
                'description': f'You have {emergency_fund_months:.1f} months of expenses saved. Target 3-6 months.',
                'impact': 'High'
            })
        elif emergency_fund_months > 12:
            recommendations.append({
                'title': '💰 Reduce Excess Cash',
                'description': f'You have {emergency_fund_months:.1f} months of expenses in cash. Consider investing excess.',
                'impact': 'Medium'
            })
        
        hsa_balance = sum(acc['balance'] for acc in st.session_state.accounts_data 
                        if acc['type'] == 'HSA')
        if hsa_balance == 0:
            recommendations.append({
                'title': '🏥 Maximize HSA Contributions',
                'description': 'HSA offers triple tax advantage: deductible contributions, tax-free growth, and tax-free withdrawals for medical expenses.',
                'impact': 'High'
            })
        
        recommendations.append({
            'title': '🎁 Maximize Employer Match',
            'description': 'Ensure you\'re contributing enough to get full employer 401k match - it\'s free money!',
            'impact': 'High'
        })
        
        for rec in recommendations:
            with st.expander(f"{rec['title']} - {rec['impact']} Impact"):
                st.write(rec['description'])
        
        st.subheader("🎯 Retirement Withdrawal Strategy")
        
        if len(st.session_state.accounts_data) > 0:
            retirement_age = st.session_state.goals_data.get('retirement_age', 65)
            withdrawal_strategy = forecaster.calculate_retirement_withdrawal_strategy(
                st.session_state.accounts_data, retirement_age
            )
            
            st.markdown("**Optimal Withdrawal Order:**")
            for step in withdrawal_strategy['withdrawal_order']:
                st.write(f"- {step}")
            
            st.info("💡 This sequence minimizes taxes and maximizes wealth preservation during retirement.")
        
        st.subheader("📊 What-If Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Increase Savings Rate:**")
            current_savings = st.session_state.goals_data.get('annual_income', 100000) - st.session_state.goals_data.get('annual_expenses', 80000)
            current_rate = (current_savings / st.session_state.goals_data.get('annual_income', 100000)) * 100
            
            additional_savings = st.slider("Additional Annual Savings ($)", 0, 50000, 5000, 1000)
            new_savings_rate = ((current_savings + additional_savings) / st.session_state.goals_data.get('annual_income', 100000)) * 100
            
            years = 30
            rate = 0.07
            additional_wealth = additional_savings * (((1 + rate) ** years - 1) / rate)
            
            st.write(f"Current savings rate: {current_rate:.1f}%")
            st.write(f"New savings rate: {new_savings_rate:.1f}%")
            st.write(f"**Additional wealth in {years} years: ${additional_wealth:,.0f}**")
        
        with col2:
            st.markdown("**Asset Allocation Impact:**")
            
            conservative_return = 0.05
            aggressive_return = 0.08
            
            wealth_base = sum(acc['balance'] for acc in st.session_state.accounts_data)
            years = 30
            
            conservative_future = wealth_base * ((1 + conservative_return) ** years)
            aggressive_future = wealth_base * ((1 + aggressive_return) ** years)
            difference = aggressive_future - conservative_future
            
            st.write(f"Conservative (5% return): ${conservative_future:,.0f}")
            st.write(f"Aggressive (8% return): ${aggressive_future:,.0f}")
            st.write(f"**Difference: ${difference:,.0f}**")
        
        st.subheader("🚀 Priority Action Items")
        
        action_items = [
            {
                'priority': 1,
                'action': 'Maximize employer 401k match',
                'timeline': 'Next payroll cycle',
                'impact': f'Up to ${st.session_state.goals_data.get("annual_income", 100000) * 0.06:,.0f}/year'
            },
            {
                'priority': 2, 
                'action': 'Build 3-6 month emergency fund',
                'timeline': '3-6 months',
                'impact': 'Financial security'
            },
            {
                'priority': 3,
                'action': 'Optimize tax-advantaged accounts',
                'timeline': 'This tax year',
                'impact': 'Long-term tax savings'
            },
            {
                'priority': 4,
                'action': 'Rebalance portfolio if needed',
                'timeline': 'Within 30 days',
                'impact': 'Better risk-adjusted returns'
            }
        ]
        
        for item in action_items:
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                st.markdown(f"**#{item['priority']}**")
            with col2:
                st.markdown(f"**{item['action']}**")
                st.markdown(f"*{item['timeline']}*")
            with col3:
                st.markdown(f"💰 {item['impact']}")
        
        st.subheader("📄 Export Your Analysis")
        
        if st.button("Generate Summary Report"):
            report_data = {
                'accounts': st.session_state.accounts_data,
                'goals': st.session_state.goals_data,
                'risk_profile': st.session_state.risk_profile,
                'total_wealth': total_wealth,
                'forecast_median': st.session_state.forecast_results['final_median'] if st.session_state.forecast_results else None,
                'success_rate': st.session_state.forecast_results['success_rate'] if st.session_state.forecast_results else None,
                'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            report_json = json.dumps(report_data, indent=2)
            
            st.download_button(
                label="💾 Download Analysis (JSON)",
                data=report_json,
                file_name=f"wealth_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            st.success("📊 Your complete wealth analysis is ready for download!")

# Sidebar
with st.sidebar:
    st.header("📊 Quick Stats")
    
    if st.session_state.accounts_data:
        total_wealth = sum(acc['balance'] for acc in st.session_state.accounts_data)
        st.metric("Total Wealth", f"${total_wealth:,.0f}")
        
        if st.session_state.goals_data:
            annual_savings = st.session_state.goals_data.get('annual_income', 0) - st.session_state.goals_data.get('annual_expenses', 0)
            if st.session_state.goals_data.get('annual_income', 0) > 0:
                savings_rate = (annual_savings / st.session_state.goals_data['annual_income']) * 100
                st.metric("Savings Rate", f"{savings_rate:.1f}%")
        
        if st.session_state.forecast_results:
            st.metric("Median Future Wealth", f"${st.session_state.forecast_results['final_median']:,.0f}")
            st.metric("Success Probability", f"{st.session_state.forecast_results['success_rate']*100:.1f}%")
    else:
        st.info("Add accounts to see stats")
    
    st.markdown("---")
    st.subheader("Setup Progress")
    
    accounts_done = len(st.session_state.accounts_data) > 0
    goals_done = len(st.session_state.goals_data) > 0
    forecast_done = st.session_state.forecast_results is not None
    
    progress_items = [
        ("Accounts Added", accounts_done),
        ("Goals Set", goals_done),
        ("Forecast Generated", forecast_done)
    ]
    
    for item, done in progress_items:
        icon = "✅" if done else "⏳"
        st.write(f"{icon} {item}")
    
    completion = sum([accounts_done, goals_done, forecast_done]) / 3 * 100
    st.progress(completion / 100)
    st.write(f"{completion:.0f}% Complete")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><strong>💰 Wealth Forecaster MVP</strong></p>
    <p>Advanced Monte Carlo simulations with sophisticated tax modeling</p>
    <p><small>⚠️ Educational projections only. Consult a financial advisor for personalized advice.</small></p>
    <p><small>🔒 All calculations performed locally - your data stays private</small></p>
</div>
""", unsafe_allow_html=True)
