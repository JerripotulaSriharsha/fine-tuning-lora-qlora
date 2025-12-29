from llama_cpp import Llama
from credit_risk_formatter import format_credit_risk_input

def load_base_model():
	"""
	Load the base Qwen2.5-3B-Instruct FP16 model
	"""
	llm = Llama(
		model_path=r"D:\Narwal\fine-tuning-lora-qlora\qwen2.5-3b-instruct-q8_0.gguf",
		n_ctx=2048,
		n_threads=8,
		n_batch=512,
		verbose=False
	)
	return llm

def ask_financial_risk_base(question: str, llm=None, streamlit_container=None):
	"""
	Ask financial risk question using base model
	"""
	if llm is None:
		llm = load_base_model()
	prompt = f"""You are a financial risk analysis assistant.\nRespond in the following format:\n<reasoning>\n(your reasoning here)\n</reasoning>\n<answer>\nChoose exactly one of: \"Good\", \"Bad\", or \"Standard\"\n</answer>\n\n{question}\n"""
	full = ""
	for chunk in llm.create_completion(prompt, max_tokens=256, stream=True):
		text = chunk["choices"][0]["text"]
		if streamlit_container:
			escaped_text = (full + text).replace('<', '&lt;').replace('>', '&gt;')
			streamlit_container.markdown(f'<div class="streaming-text">{escaped_text}</div>', unsafe_allow_html=True)
		else:
			print(text, end="", flush=True)
		full += text
	return full

if __name__ == "__main__":
	# Test the base model
	llm = load_base_model()
	print("Enter the formatted input string (output from credit_risk_formatter.py):")
	test_input = input("Formatted Input: ")
	print(f"\nUsing Input: {test_input}")
	result = ask_financial_risk_base(test_input)
	print(f"\n\nBase Model Result:\n{result}")
