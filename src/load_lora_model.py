from llama_cpp import Llama
from credit_risk_formatter import format_credit_risk_input

def load_lora_model():
    """
    Load the LoRA fine-tuned model
    """
    llm = Llama(
        model_path=r"D:\Narwal\fine-tuning-lora-qlora\qwen2.5-3b--lora-f16.gguf",
        n_ctx=2048,
        n_threads=8,
        n_batch=512,
        verbose=False
    )
    return llm

def ask_financial_risk_lora(question: str, llm=None, streamlit_container=None):
    """
    Ask financial risk question using LoRA model
    """
    if llm is None:
        llm = load_lora_model()
    
    prompt = f"""You are a financial risk analysis assistant.
Respond in the following format:
<reasoning>
(your reasoning here)
</reasoning>
<answer>
Choose exactly one of: "Good", "Bad", or "Standard"
</answer>

{question}
"""
    full = ""
    for chunk in llm.create_completion(prompt, max_tokens=256, stream=True):
        text = chunk["choices"][0]["text"]
        if streamlit_container:
            # Escape HTML characters to prevent JavaScript errors
            escaped_text = (full + text).replace('<', '&lt;').replace('>', '&gt;')
            streamlit_container.markdown(f'<div class="streaming-text">{escaped_text}</div>', unsafe_allow_html=True)
        else:
            print(text, end="", flush=True)
        full += text
    return full

if __name__ == "__main__":
    # Test the LoRA model
    llm = load_lora_model()
    
    # Get the formatted input string (output from credit_risk_formatter.py)
    print("Enter the formatted input string (output from credit_risk_formatter.py):")
    test_input = input("Formatted Input: ")
    
    print(f"\nUsing Input: {test_input}")
    result = ask_financial_risk_lora(test_input)
    print(f"\n\nLoRA Model Result:\n{result}")
