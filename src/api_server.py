from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from credit_risk_formatter import format_credit_risk_input
from load_qlora_model import load_qlora_model, ask_financial_risk_qlora
from load_lora_model import load_lora_model, ask_financial_risk_lora

# Global variables to store loaded models
qlora_model = None
lora_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown"""
    global qlora_model, lora_model
    print("Loading models...")
    qlora_model = load_qlora_model()
    lora_model = load_lora_model()
    print("Models loaded successfully!")
    yield
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="API for running QLoRA and LoRA models for credit risk assessment",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CreditRiskRequest(BaseModel):
    age: int
    occupation: str
    annual_income: float
    outstanding_debt: float
    credit_utilization: float
    payment_behavior: str

class ModelResponse(BaseModel):
    model_name: str
    formatted_input: str
    response: str
    processing_time: float

class ParallelResponse(BaseModel):
    qlora_result: ModelResponse
    lora_result: ModelResponse
    total_processing_time: float


def run_qlora_inference(request_data: CreditRiskRequest) -> ModelResponse:
    """Run QLoRA model inference"""
    import time
    start_time = time.time()
    
    # Format the input
    formatted_input = format_credit_risk_input(
        age=request_data.age,
        occupation=request_data.occupation,
        annual_income=request_data.annual_income,
        credit_utilization=request_data.credit_utilization,
        outstanding_debt=request_data.outstanding_debt,
        payment_behavior=request_data.payment_behavior,
        credit_mix="Standard"
    )
    
    # Get model response
    response = ask_financial_risk_qlora(formatted_input, qlora_model)
    
    processing_time = time.time() - start_time
    
    return ModelResponse(
        model_name="QLoRA",
        formatted_input=formatted_input,
        response=response,
        processing_time=processing_time
    )

def run_lora_inference(request_data: CreditRiskRequest) -> ModelResponse:
    """Run LoRA model inference"""
    import time
    start_time = time.time()
    
    # Format the input
    formatted_input = format_credit_risk_input(
        age=request_data.age,
        occupation=request_data.occupation,
        annual_income=request_data.annual_income,
        credit_utilization=request_data.credit_utilization,
        outstanding_debt=request_data.outstanding_debt,
        payment_behavior=request_data.payment_behavior,
        credit_mix="Standard"
    )
    
    # Get model response
    response = ask_financial_risk_lora(formatted_input, lora_model)
    
    processing_time = time.time() - start_time
    
    return ModelResponse(
        model_name="LoRA",
        formatted_input=formatted_input,
        response=response,
        processing_time=processing_time
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "qlora": "/inference/qlora",
            "lora": "/inference/lora", 
            "parallel": "/inference/parallel",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": qlora_model is not None and lora_model is not None
    }

@app.post("/inference/qlora", response_model=ModelResponse)
async def qlora_inference(request: CreditRiskRequest):
    """Run QLoRA model inference"""
    if qlora_model is None:
        raise HTTPException(status_code=500, detail="QLoRA model not loaded")
    
    try:
        result = run_qlora_inference(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QLoRA inference failed: {str(e)}")

@app.post("/inference/lora", response_model=ModelResponse)
async def lora_inference(request: CreditRiskRequest):
    """Run LoRA model inference"""
    if lora_model is None:
        raise HTTPException(status_code=500, detail="LoRA model not loaded")
    
    try:
        result = run_lora_inference(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LoRA inference failed: {str(e)}")

@app.post("/inference/parallel", response_model=ParallelResponse)
async def parallel_inference(request: CreditRiskRequest):
    """Run both models in parallel"""
    if qlora_model is None or lora_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    import time
    start_time = time.time()
    
    try:
        # Run both models in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            qlora_future = executor.submit(run_qlora_inference, request)
            lora_future = executor.submit(run_lora_inference, request)
            
            # Wait for both to complete
            qlora_result = qlora_future.result()
            lora_result = lora_future.result()
        
        total_time = time.time() - start_time
        
        return ParallelResponse(
            qlora_result=qlora_result,
            lora_result=lora_result,
            total_processing_time=total_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parallel inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
