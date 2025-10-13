import argparse
import json
import os
from typing import Dict

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

from credit_risk_formatter import format_credit_risk_input


def parse_feature_string(s: str) -> Dict[str, str]:
    parts = [p.strip() for p in s.split("|")]
    result: Dict[str, str] = {}
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def build_prompt_from_features(feat: Dict[str, str]) -> str:
    age = int(float(feat.get("Age", 0)))
    occupation = feat.get("Occupation", "Unknown")
    annual_income = float(feat.get("Annual_Income", 0.0))
    credit_utilization = float(feat.get("Credit_Utilization_Ratio", 0.0))
    outstanding_debt = float(feat.get("Outstanding_Debt", 0.0))
    payment_behavior = feat.get("Payment_Behaviour", feat.get("Payment_Behavior", "Unknown"))
    credit_mix = feat.get("Credit_Mix", "Standard")
    return format_credit_risk_input(
        age=age,
        occupation=occupation,
        annual_income=annual_income,
        credit_utilization=credit_utilization,
        outstanding_debt=outstanding_debt,
        payment_behavior=payment_behavior,
        credit_mix=credit_mix,
    )


def load_model(adapter_dir: str):
    base_model = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"
    max_seq_length = 4096
    dtype = None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_dir)
    # Merge for faster inference if possible
    try:
        model = model.merge_and_unload()
    except Exception:
        pass

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.3, top_p: float = 0.9) -> str:
    messages = [
        {"role": "system", "content": "You are a senior credit risk analyst. Provide clear, compliant, and well-reasoned assessments."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference with LoRA adapter from outputs/checkpoint-250")
    parser.add_argument("--adapter_dir", default=os.path.join("outputs", "checkpoint-250"))
    parser.add_argument("--features", required=False, help="Pipe-delimited key:value string of features")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_dir)

    if args.features:
        feat = parse_feature_string(args.features)
        prompt = build_prompt_from_features(feat)
    else:
        # Fallback example
        prompt = build_prompt_from_features({
            "Age": "32",
            "Occupation": "Journalist",
            "Annual_Income": "33470.43",
            "Credit_Utilization_Ratio": "26.8",
            "Outstanding_Debt": "1318.49",
            "Payment_Behaviour": "High_spent_Small_value_payments",
            "Credit_Mix": "Standard",
        })

    answer = generate_answer(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== MODEL ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()



