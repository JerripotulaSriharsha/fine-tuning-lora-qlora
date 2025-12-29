# Credit Risk Assessment System - Design Document

**Version:** 1.0.0
**Date:** November 2025
**Project:** Fine-Tuning LoRA/QLoRA for Credit Risk Classification
**Author:** AI/ML Engineering Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Data Architecture](#data-architecture)
4. [Model Architecture](#model-architecture)
5. [User Interface Design](#user-interface-design)
6. [API Design](#api-design)
7. [Training Pipeline Design](#training-pipeline-design)
8. [Deployment Architecture](#deployment-architecture)
9. [Security and Privacy](#security-and-privacy)
10. [Performance Considerations](#performance-considerations)

---

## 1. Executive Summary

### 1.1 Project Overview

The Credit Risk Assessment System is an AI-powered solution that leverages parameter-efficient fine-tuning techniques (LoRA and QLoRA) to classify customers into three credit risk categories: **Good**, **Bad**, and **Standard**. The system is built on top of the Qwen2.5-3B-Instruct language model and provides multiple deployment interfaces including a web UI, REST API, and direct inference scripts.

### 1.2 Design Goals

- **Accuracy**: Achieve >50% classification accuracy compared to 20% baseline
- **Efficiency**: Enable training on consumer-grade GPUs (8GB VRAM)
- **Accessibility**: Provide multiple interfaces (Web UI, API, CLI)
- **Scalability**: Support parallel model comparison and batch inference
- **Maintainability**: Modular architecture with clear separation of concerns

### 1.3 Key Innovations

- **GRPO Training**: Group Relative Policy Optimization with multi-objective reward functions
- **Parameter Efficiency**: 4-bit quantization enabling 3B model training on 8GB GPU
- **Multi-Model Comparison**: Side-by-side evaluation of Base, LoRA, and QLoRA variants
- **Production Ready**: Complete deployment stack with streaming inference

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Web UI (Streamlit)  │  REST API Client  │  CLI Tools           │
└──────────┬──────────────────┬────────────────────┬──────────────┘
           │                  │                    │
           v                  v                    v
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  app.py             │  api_server.py     │  infer.py            │
│  (Streamlit)        │  (FastAPI)         │  (CLI)               │
└──────────┬──────────────────┬────────────────────┬──────────────┘
           │                  │                    │
           v                  v                    v
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  load_qlora_model.py  │  load_lora_model.py  │ load_base_model.py│
│  (GGUF Loaders)       │  (GGUF Loaders)      │ (GGUF Loaders)   │
└──────────┬──────────────────┬────────────────────┬──────────────┘
           │                  │                    │
           v                  v                    v
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  qwen2.5-3b-f16-qlora.gguf  │  qwen2.5-3b--lora-f16.gguf       │
│  (6.18 GB)                  │  (6.18 GB)                        │
│  qwen2.5-3b-instruct-q8_0.gguf  (3.61 GB)                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│  train_qlora.ipynb  │  train_sft_qlora.ipynb  │  trainlora.ipynb│
│  (GRPO Training)    │  (SFT Training)         │  (Full LoRA)    │
└──────────┬──────────────────┬────────────────────┬──────────────┘
           │                  │                    │
           v                  v                    v
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  creditmix_dataset.json (31,868 examples)                       │
│  evaluation_examples.json (10 test cases)                       │
│  test.csv (50,000 raw records)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Web UI** | Interactive credit risk assessment | Streamlit |
| **API Server** | RESTful inference endpoints | FastAPI |
| **Model Loaders** | Model loading and inference orchestration | llama.cpp, Python |
| **Training Pipeline** | Model fine-tuning and evaluation | Unsloth, TRL, PEFT |
| **Data Processor** | Input formatting and validation | Python |

---

## 3. Data Architecture

### 3.1 Input Schema

**Customer Financial Profile**
```python
{
    "age": int,                          # 18-100 years
    "occupation": str,                   # Job title
    "annual_income": float,              # USD, positive
    "outstanding_debt": float,           # USD, positive
    "credit_utilization": float,         # 0-100%
    "payment_behavior": str              # Enum of 6 categories
}
```

**Payment Behavior Categories:**
1. `Low_spent_Small_value_payments`
2. `High_spent_Medium_value_payments`
3. `Low_spent_Medium_value_payments`
4. `Low_spent_Large_value_payments`
5. `High_spent_Large_value_payments`
6. `High_spent_Small_value_payments`

### 3.2 Output Schema

**Credit Risk Classification**
```xml
<reasoning>
[Model's analytical reasoning about the customer's financial profile]
</reasoning>
<answer>
[One of: "Good", "Bad", "Standard"]
</answer>
```

### 3.3 Training Dataset Structure

**File:** `creditmix_dataset.json`

```json
{
  "question": "Age: X, Occupation: Y, Annual Income: Z, Outstanding Debt: W, Credit Utilization Ratio: V, Payment Behaviour: U",
  "answer": "Good|Bad|Standard"
}
```

**Dataset Statistics:**
- Total examples: 31,868
- Balanced version: 22,764 (7,588 per class)
- Distribution:
  - Good: 30.4% (9,685 examples)
  - Standard: 45.8% (14,595 examples)
  - Bad: 23.8% (7,588 examples)

### 3.4 Data Flow

```
Raw CSV (50K rows)
    ↓
Preprocessing & Cleaning
    ↓
JSON Format Conversion
    ↓
Class Balancing
    ↓
HuggingFace Upload (Sri1999/creditmix-dataset)
    ↓
Training Pipeline
    ↓
Model Checkpoints
    ↓
GGUF Conversion
    ↓
Inference Deployment
```

---

## 4. Model Architecture

### 4.1 Base Model

**Model:** Qwen/Qwen2.5-3B-Instruct

**Specifications:**
- Parameters: 3 billion
- Architecture: Transformer-based decoder
- Vocabulary: 151,936 tokens
- Context Length: 32,768 tokens
- Quantization: 4-bit (NF4) for training, 8-bit/16-bit for inference

### 4.2 LoRA Configuration

**Rank (r):** 32
**Alpha:** 64
**Dropout:** 0.0
**Target Modules:**
- `q_proj` (Query projection)
- `k_proj` (Key projection)
- `v_proj` (Value projection)
- `o_proj` (Output projection)
- `gate_proj` (Gate projection for MLP)
- `up_proj` (Up projection for MLP)
- `down_proj` (Down projection for MLP)

**Trainable Parameters:**
- LoRA adapters: ~119 MB
- Total model: 3B parameters
- Adapter percentage: ~1.3% of total parameters

### 4.3 Model Variants

| Model | Quantization | Size | Accuracy | Use Case |
|-------|-------------|------|----------|----------|
| **Base** | 8-bit | 3.61 GB | 20% | Baseline comparison |
| **QLoRA** | 4-bit + LoRA | 6.18 GB | 50% | Memory-efficient fine-tuning |
| **LoRA** | 16-bit + LoRA | 6.18 GB | 60% | Best accuracy |

### 4.4 Inference Configuration

**Generation Parameters:**
```python
{
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "stop_sequences": ["</answer>", "\n\n"]
}
```

**Streaming:** Enabled for real-time token generation

---

## 5. User Interface Design

### 5.1 Web UI (Streamlit)

**Layout Structure:**
```
┌─────────────────────────────────────────────┐
│         🏦 Credit Risk Assessment Tool       │
├─────────────────────────────────────────────┤
│  📋 Customer Information                     │
│  ┌─────────────┬─────────────┐              │
│  │ Age         │ Outstanding │              │
│  │ Occupation  │ Debt        │              │
│  │ Income      │ Credit Util │              │
│  │             │ Payment Beh │              │
│  └─────────────┴─────────────┘              │
├─────────────────────────────────────────────┤
│  📊 Model Performance                        │
│  ┌──────────┬──────────┬──────────┐         │
│  │  Base    │  LoRA    │  QLoRA   │         │
│  │  20%     │  60%     │  50%     │         │
│  └──────────┴──────────┴──────────┘         │
├─────────────────────────────────────────────┤
│  🤖 Select Models to Compare                │
│  ☑ QLoRA  ☐ LoRA  ☐ Base                   │
├─────────────────────────────────────────────┤
│  🔍 Generate Assessment                     │
│  [Generate Credit Risk Assessment]          │
├─────────────────────────────────────────────┤
│  Results:                                    │
│  ┌─────────────────────────────────────┐    │
│  │ QLoRA Model                         │    │
│  │ <streaming output...>               │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Design Principles:**

1. **Visual Hierarchy**: Clear separation between input, configuration, and results
2. **Color Coding**: Distinct gradient backgrounds for each model
   - QLoRA: Blue gradient (#1e3c72 → #2a5298)
   - LoRA: Orange gradient (#ff6b35 → #f7931e)
   - Base: Purple gradient (#764ba2 → #667eea)
3. **Responsive Layout**: Two-column grid for input fields
4. **Real-time Feedback**: Streaming text display with monospace font
5. **Transparency**: Model performance metrics displayed prominently

### 5.2 Color Palette

| Element | Primary | Secondary | Text |
|---------|---------|-----------|------|
| **QLoRA Container** | #1e3c72 | #2a5298 | #ffffff |
| **LoRA Container** | #ff6b35 | #f7931e | #ffffff |
| **Base Container** | #764ba2 | #667eea | #ffffff |
| **Input Section** | #f5f7fa | #c3cfe2 | #2c3e50 |
| **Buttons** | #ff4b4b | #e03e3e | #ffffff |

### 5.3 Typography

- **Headers**: Bold, 2.5rem (main), 1.5rem (sections)
- **Body Text**: Regular, 1rem
- **Model Titles**: Bold, 1.3rem, white with shadow
- **Streaming Output**: Monospace (Courier New), 0.9rem

---

## 6. API Design

### 6.1 REST API Endpoints

**Base URL:** `http://localhost:8000`

#### 6.1.1 Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

#### 6.1.2 QLoRA Inference
```
POST /inference/qlora
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 32,
  "occupation": "Journalist",
  "annual_income": 33470.43,
  "outstanding_debt": 1318.49,
  "credit_utilization": 26.8,
  "payment_behavior": "High_spent_Small_value_payments"
}
```

**Response:**
```json
{
  "model_name": "QLoRA",
  "formatted_input": "Age: 32, Occupation: Journalist, ...",
  "response": "<reasoning>...</reasoning><answer>Good</answer>",
  "processing_time": 2.45
}
```

#### 6.1.3 LoRA Inference
```
POST /inference/lora
Content-Type: application/json
```
(Same request/response structure as QLoRA)

#### 6.1.4 Parallel Inference
```
POST /inference/parallel
Content-Type: application/json
```

**Response:**
```json
{
  "qlora_result": { /* ModelResponse */ },
  "lora_result": { /* ModelResponse */ },
  "total_processing_time": 3.12
}
```

### 6.2 API Features

- **CORS Enabled**: Cross-origin requests allowed
- **Model Preloading**: Models loaded at startup for fast inference
- **Parallel Processing**: ThreadPoolExecutor for concurrent model runs
- **Error Handling**: Structured HTTP exceptions
- **Auto-Documentation**: OpenAPI/Swagger at `/docs`

---

## 7. Training Pipeline Design

### 7.1 Training Strategy

**Method:** GRPO (Group Relative Policy Optimization)

**Rationale:**
- Traditional supervised fine-tuning only teaches the model to mimic outputs
- GRPO uses reinforcement learning to optimize for multiple objectives
- Reward functions guide the model to produce correct format AND correct classification

### 7.2 Multi-Objective Reward Function

```python
def reward_function(prompt, response, ground_truth):
    rewards = []

    # 1. XML Format Validation
    if contains_xml_tags(response):
        rewards.append(1.0)

    # 2. Soft Format Matching
    if has_reasoning_and_answer(response):
        rewards.append(0.5)

    # 3. Strict Format Matching
    if exact_xml_format(response):
        rewards.append(1.0)

    # 4. Category Validation
    if answer_in_valid_categories(response):
        rewards.append(1.0)

    # 5. Correctness Reward
    if extracted_answer == ground_truth:
        rewards.append(2.0)  # Highest weight
    else:
        rewards.append(0.0)

    return sum(rewards)
```

### 7.3 Training Configuration

**Optimizer:** paged_adamw_8bit
**Learning Rate:** 5e-6
**Scheduler:** Cosine annealing
**Batch Size:** 6 per device
**Gradient Accumulation:** 1 step
**Max Steps:** 100
**Warmup Steps:** 5
**Checkpointing:** Every 500 steps (18 checkpoints)

**Hardware:**
- GPU: NVIDIA RTX 4060 (8GB VRAM)
- Memory Optimization: 4-bit quantization + gradient checkpointing

### 7.4 Training Flow

```
Load Base Model (4-bit quantized)
    ↓
Initialize LoRA Adapters
    ↓
Load Balanced Dataset (22,764 examples)
    ↓
GRPO Training Loop (100 steps)
    ├── Generate responses
    ├── Calculate multi-objective rewards
    ├── Update policy
    └── Save checkpoint every 500 steps
    ↓
Merge Adapters with Base Model
    ↓
Convert to GGUF Format
    ↓
Upload to HuggingFace Hub
```

### 7.5 Evaluation Pipeline

**Test Set:** 10 examples (4 Good, 3 Standard, 3 Bad)

**Metrics:**
- Accuracy: Percentage of correct classifications
- Per-class Precision/Recall
- Format Compliance Rate
- Response Quality (human evaluation)

**Results:**
- Base Model: 2/10 = 20%
- QLoRA Model: 5/10 = 50%
- LoRA Model: 6/10 = 60%

---

## 8. Deployment Architecture

### 8.1 Deployment Options

#### Option 1: Streamlit Web App
```bash
streamlit run src/app.py --server.port 8501
```
- Best for: Interactive demos, internal tools
- Users: Non-technical stakeholders, QA testers

#### Option 2: FastAPI Server
```bash
python src/api_server.py
```
- Best for: Production API, system integrations
- Users: Backend services, mobile apps, web clients

#### Option 3: CLI Inference
```bash
python src/infer.py --checkpoint outputs/checkpoint-8500
```
- Best for: Batch processing, scripting, automation
- Users: Data scientists, DevOps engineers

### 8.2 Containerization (Recommended)

**Dockerfile Structure:**
```dockerfile
FROM python:3.10-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy models and code
COPY models/ /app/models/
COPY src/ /app/src/

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0"]
```

### 8.3 Scaling Strategy

**Horizontal Scaling:**
- Load balancer (NGINX/Traefik)
- Multiple API server instances
- Shared model storage (NFS/S3)

**Vertical Scaling:**
- Upgrade to GPU instances for faster inference
- Increase memory for larger batch sizes
- Use model quantization (INT8, INT4) for lower latency

### 8.4 Monitoring and Logging

**Metrics to Track:**
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Model accuracy (online evaluation)
- Error rates
- GPU/CPU utilization
- Memory usage

**Tools:**
- Prometheus + Grafana for metrics
- ELK Stack for log aggregation
- Sentry for error tracking

---

## 9. Security and Privacy

### 9.1 Data Privacy

- **PII Handling**: No personally identifiable information stored
- **Data Retention**: API requests not logged by default
- **Encryption**: HTTPS for all API communications
- **Anonymization**: Input data contains no names, addresses, or SSNs

### 9.2 Model Security

- **Model Signing**: Verify model integrity with checksums
- **Access Control**: API key authentication (to be implemented)
- **Rate Limiting**: Prevent abuse (to be implemented)
- **Input Validation**: Strict schema validation prevents injection attacks

### 9.3 Compliance Considerations

- **GDPR**: Right to explanation (model provides reasoning)
- **Fair Lending**: Monitor for bias in credit decisions
- **Audit Trail**: Log all predictions with timestamps
- **Explainability**: XML reasoning format provides transparency

---

## 10. Performance Considerations

### 10.1 Latency Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| **Model Loading** | < 30s | ~15s |
| **Single Inference** | < 5s | ~2-3s |
| **Parallel Inference** | < 7s | ~3-4s |
| **Streaming First Token** | < 500ms | ~300ms |

### 10.2 Optimization Techniques

1. **Model Quantization**: 4-bit/8-bit reduces memory and increases speed
2. **GGUF Format**: Optimized for CPU inference via llama.cpp
3. **Model Caching**: Load once at startup, reuse for all requests
4. **Parallel Execution**: ThreadPoolExecutor for concurrent model runs
5. **Streaming Output**: Tokens generated incrementally for better UX

### 10.3 Resource Requirements

**Development:**
- CPU: 4 cores minimum
- RAM: 16 GB
- GPU: 8 GB VRAM (for training)
- Storage: 20 GB

**Production (API Server):**
- CPU: 8 cores recommended
- RAM: 32 GB
- GPU: Optional (16 GB for faster inference)
- Storage: 30 GB

### 10.4 Bottlenecks and Mitigations

| Bottleneck | Impact | Mitigation |
|------------|--------|------------|
| **Model Size** | Slow loading | Use quantized GGUF format |
| **Single Thread** | Low throughput | Parallel inference endpoints |
| **Memory** | OOM errors | Gradient checkpointing, 4-bit quantization |
| **I/O** | Disk latency | Load models into RAM, use SSD |

---

## Appendix A: Technology Stack

### Core ML/AI
- **PyTorch**: 2.4.0+cu121
- **Transformers**: 4.55.4
- **Unsloth**: Latest (training optimization)
- **TRL**: Latest (reinforcement learning)
- **PEFT**: Latest (parameter-efficient fine-tuning)
- **bitsandbytes**: 0.41.0+ (quantization)
- **llama.cpp**: Latest (GGUF inference)

### Web Framework
- **Streamlit**: 1.28.0+ (UI)
- **FastAPI**: 0.104.1 (API)
- **Uvicorn**: 0.24.0 (ASGI server)

### Data Processing
- **Pandas**: Latest
- **NumPy**: Latest
- **Datasets**: HuggingFace datasets library

### DevOps
- **Git**: Version control
- **Docker**: Containerization
- **Python**: 3.10+

---

## Appendix B: File Structure

```
fine-tuning-lora-qlora/
├── src/
│   ├── app.py                      # Streamlit web UI
│   ├── api_server.py               # FastAPI REST API
│   ├── infer.py                    # CLI inference script
│   ├── load_qlora_model.py         # QLoRA model loader
│   ├── load_lora_model.py          # LoRA model loader
│   ├── load_base_model.py          # Base model loader
│   ├── credit_risk_formatter.py    # Input formatter
│   ├── requirements.txt            # Core dependencies
│   └── api_requirements.txt        # API dependencies
├── outputs/
│   ├── checkpoint-500/             # Training checkpoints
│   ├── checkpoint-1000/
│   └── ...
├── train_qlora.ipynb               # GRPO training notebook
├── train_sft_qlora.ipynb           # SFT training notebook
├── trainlora.ipynb                 # Full LoRA training
├── evaluation.ipynb                # Model evaluation
├── preprocessing.ipynb             # Data preprocessing
├── creditmix_dataset.json          # Training dataset
├── evaluation_examples.json        # Test dataset
├── qwen2.5-3b-f16-qlora.gguf       # QLoRA model (6.18 GB)
├── qwen2.5-3b--lora-f16.gguf       # LoRA model (6.18 GB)
└── qwen2.5-3b-instruct-q8_0.gguf   # Base model (3.61 GB)
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Nov 2025 | AI/ML Team | Initial design document |

---

**End of Design Document**
