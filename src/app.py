import streamlit as st
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from credit_risk_formatter import format_credit_risk_input
from load_qlora_model import load_qlora_model, ask_financial_risk_qlora
from load_lora_model import load_lora_model, ask_financial_risk_lora
from load_base_model import load_base_model, ask_financial_risk_base

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* ===== BUTTON STYLING ===== */
    .stButton>button {
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
        background-color: #ff4b4b !important;  /* red theme */
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #e03e3e !important;
        transform: scale(1.02);
    }

    .stButton>button:focus, .stButton>button:active {
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }

    /* ===== HEADER + SECTIONS ===== */
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

    /* ===== MODEL CONTAINERS ===== */
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

    .base-container {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
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
    """Load all models and cache them. If base model file is missing, return None for base_model."""
    qlora_model = load_qlora_model()
    lora_model = load_lora_model()
    base_model_path = r"D:\Narwal\fine-tuning-lora-qlora\qwen2.5-3b-instruct-q8_0.gguf"
    if os.path.exists(base_model_path):
        base_model = load_base_model()
    else:
        base_model = None
    return qlora_model, lora_model, base_model


def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Assessment Tool</h1>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📋 Customer Information</h2>', unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns([1, 1])
    
    with input_col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=32, step=1, help="Customer's age in years"
        )
        occupation = st.text_input(
            "Occupation", value="Journalist", help="Customer's current occupation"
        )
        annual_income = st.number_input(
            "Annual Income ($)", min_value=0.0, value=33470.43, step=100.0, format="%.2f",
            help="Customer's annual income in USD"
        )
    
    with input_col2:
        outstanding_debt = st.number_input(
            "Outstanding Debt ($)", min_value=0.0, value=1318.49, step=10.0, format="%.2f",
            help="Total outstanding debt amount in USD"
        )
        credit_utilization = st.number_input(
            "Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=26.8,
            step=0.1, format="%.1f", help="Credit utilization ratio as a percentage"
        )
        payment_behaviors = [
            'Low_spent_Small_value_payments',
            'High_spent_Medium_value_payments',
            'Low_spent_Medium_value_payments',
            'Low_spent_Large_value_payments',
            'High_spent_Large_value_payments',
            'High_spent_Small_value_payments'
        ]
        payment_behavior = st.selectbox(
            "Payment Behaviour", options=payment_behaviors, index=3,
            help="Customer's payment behavior pattern"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Performance Section
    st.markdown('<h2 class="section-header">📊 Model Performance (10 Sample Evaluation)</h2>', unsafe_allow_html=True)
    
    # Create columns for accuracy display
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    
    with acc_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
            <h3 style="margin: 0; font-size: 1.2rem;">Base Model</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">20%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Baseline Performance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with acc_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
            <h3 style="margin: 0; font-size: 1.2rem;">LoRA Model</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">60%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">+40% Improvement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with acc_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
            <h3 style="margin: 0; font-size: 1.2rem;">QLoRA Model</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">50%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">+30% Improvement</div>
        </div>
        """, unsafe_allow_html=True)

    # Model selection
    st.markdown('<h2 class="section-header">🤖 Select Models to Compare</h2>', unsafe_allow_html=True)
    
    # Create checkboxes for multiple model selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qlora_selected = st.checkbox("QLoRA", value=True, help="Select QLoRA model for comparison")
    
    with col2:
        lora_selected = st.checkbox("LoRA", value=False, help="Select LoRA model for comparison")
    
    with col3:
        base_selected = st.checkbox("Base", value=False, help="Select Base model for comparison")
    
    # Check if at least one model is selected
    selected_models = []
    if qlora_selected:
        selected_models.append("QLoRA")
    if lora_selected:
        selected_models.append("LoRA")
    if base_selected:
        selected_models.append("Base")
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model to run!")

    # Generate button
    st.markdown('<h2 class="section-header">🔍 Generate Assessment</h2>', unsafe_allow_html=True)

    if st.button("Generate Credit Risk Assessment", type="primary", use_container_width=True) and selected_models:
        formatted_input = format_credit_risk_input(
            age=int(age),
            occupation=occupation,
            annual_income=float(annual_income),
            credit_utilization=float(credit_utilization),
            outstanding_debt=float(outstanding_debt),
            payment_behavior=payment_behavior,
            credit_mix="Standard"
        )

        with st.spinner(f"Loading models: {', '.join(selected_models)}..."):
            qlora_model, lora_model, base_model = load_models()

        # Run all selected models
        for model_name in selected_models:
            # Dynamic container background
            if model_name == "QLoRA":
                container_class = "qlora-container"
            elif model_name == "LoRA":
                container_class = "lora-container"
            else:
                container_class = "base-container"

            st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            st.markdown(f'<div class="model-title">{model_name} Model</div>', unsafe_allow_html=True)
            container = st.empty()
            container.markdown(f'<div class="streaming-text">Starting {model_name} model...</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Run the selected model
            if model_name == "QLoRA":
                result = ask_financial_risk_qlora(formatted_input, qlora_model, container)
            elif model_name == "LoRA":
                result = ask_financial_risk_lora(formatted_input, lora_model, container)
            elif model_name == "Base":
                if base_model:
                    result = ask_financial_risk_base(formatted_input, base_model, container)
                else:
                    container.markdown('<div class="streaming-text">Base model file not found.</div>', unsafe_allow_html=True)

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
