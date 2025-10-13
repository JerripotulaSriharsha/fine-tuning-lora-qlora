import streamlit as st
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from credit_risk_formatter import format_credit_risk_input
from load_qlora_model import load_qlora_model, ask_financial_risk_qlora
from load_lora_model import load_lora_model, ask_financial_risk_lora

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .model-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .qlora-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(30,60,114,0.3);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .lora-container {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255,107,53,0.3);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .model-title {
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .streaming-text {
        background-color: rgba(0,0,0,0.3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.5;
        min-height: 200px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load both models and cache them"""
    qlora_model = load_qlora_model()
    lora_model = load_lora_model()
    return qlora_model, lora_model


def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Assessment Tool</h1>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    # Input section with better styling
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📋 Customer Information</h2>', unsafe_allow_html=True)
    
    # Create two columns for input fields
    input_col1, input_col2 = st.columns([1, 1])
    
    with input_col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=32,
            step=1,
            help="Customer's age in years"
        )
        
        occupation = st.text_input(
            "Occupation",
            value="Journalist",
            help="Customer's current occupation"
        )
        
        annual_income = st.number_input(
            "Annual Income ($)",
            min_value=0.0,
            value=33470.43,
            step=100.0,
            format="%.2f",
            help="Customer's annual income in USD"
        )
    
    with input_col2:
        outstanding_debt = st.number_input(
            "Outstanding Debt ($)",
            min_value=0.0,
            value=1318.49,
            step=10.0,
            format="%.2f",
            help="Total outstanding debt amount in USD"
        )
        
        credit_utilization = st.number_input(
            "Credit Utilization Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=26.8,
            step=0.1,
            format="%.1f",
            help="Credit utilization ratio as a percentage"
        )
        
        # Payment behavior dropdown
        payment_behaviors = [
            'Low_spent_Small_value_payments',
            'High_spent_Medium_value_payments',
            'Low_spent_Medium_value_payments',
            'Low_spent_Large_value_payments',
            'High_spent_Large_value_payments',
            'High_spent_Small_value_payments'
        ]
        
        payment_behavior = st.selectbox(
            "Payment Behaviour",
            options=payment_behaviors,
            index=3,  # Default to 'Low_spent_Large_value_payments'
            help="Customer's payment behavior pattern"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
        
    
    
    # Generate formatted input button
    st.markdown('<h2 class="section-header">🔍 Generate Assessment</h2>', unsafe_allow_html=True)
    
    if st.button("Generate Credit Risk Assessment", type="primary", use_container_width=True):
        # Generate the formatted input
        formatted_input = format_credit_risk_input(
            age=int(age),
            occupation=occupation,
            annual_income=float(annual_income),
            credit_utilization=float(credit_utilization),
            outstanding_debt=float(outstanding_debt),
            payment_behavior=payment_behavior,
            credit_mix="Standard"  # Default value since it's a label
        )
        
        
        # Load models
        with st.spinner("Loading models..."):
            qlora_model, lora_model = load_models()
        
        # Create two columns for model results
        st.markdown('<h2 class="section-header">🤖 Model Predictions</h2>', unsafe_allow_html=True)
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown('<div class="qlora-container">', unsafe_allow_html=True)
            st.markdown('<div class="model-title">🚀 QLoRA Model</div>', unsafe_allow_html=True)
            qlora_container = st.empty()
            qlora_container.markdown('<div class="streaming-text">Starting QLoRA model...</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col2:
            st.markdown('<div class="lora-container">', unsafe_allow_html=True)
            st.markdown('<div class="model-title">⚡ LoRA Model</div>', unsafe_allow_html=True)
            lora_container = st.empty()
            lora_container.markdown('<div class="streaming-text">Starting LoRA model...</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Run both models sequentially but with real-time updates
        # This approach works better with Streamlit's architecture
        
        # Start QLoRA model
        qlora_result = ask_financial_risk_qlora(formatted_input, qlora_model, qlora_container)
        
        # Start LoRA model
        lora_result = ask_financial_risk_lora(formatted_input, lora_model, lora_container)
        
        # Display the raw input data for reference
        with st.expander("📋 Raw Input Data"):
            st.json({
                "Age": int(age),
                "Occupation": occupation,
                "Annual Income": float(annual_income),
                "Credit Utilization Ratio": float(credit_utilization),
                "Outstanding Debt": float(outstanding_debt),
                "Payment Behaviour": payment_behavior
            })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d;'>Credit Risk Assessment Tool - Built with Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
