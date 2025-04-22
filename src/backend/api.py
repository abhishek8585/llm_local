"""
FastAPI Backend for LLM Chat Interface
This is the main backend file that handles all API endpoints and model interactions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import time
from huggingface_hub import login
import uuid
import psutil
import gc
import uvicorn

# Remove any existing handlers
logging.getLogger().handlers = []

# Create custom formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Add request_id to record if it doesn't exist
        if not hasattr(record, 'request_id'):
            if record.name.startswith('src.backend'):
                record.request_id = 'SYSTEM'
            else:
                record.request_id = 'EXTERNAL'
        
        # Use the parent class to do the actual formatting
        return super().format(record)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('api.log')

# Set formatters
formatter = CustomFormatter('%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Create our custom logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Login to Hugging Face Hub
try:
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    logger.info("Successfully logged in to Hugging Face Hub")
except Exception as e:
    logger.error(f"Failed to login to Hugging Face Hub: {e}")
    raise

app = FastAPI(title="LLM Chat API")

# Model configuration
MODELS = {
    "phi-3-mini": "microsoft/phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama-3-8b": "meta-llama/Llama-3.1-8B-Instruct"
}
DEFAULT_MODEL = "gemma-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and tokenizer
tokenizer = None
model = None
current_model_name = None

# Define request/response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class SimpleChatRequest(BaseModel):
    model: str
    message: str
    thread_id: Optional[str] = None

# Thread memory storage
thread_memory: Dict[str, List[ChatMessage]] = {}

def get_thread_messages(thread_id: str) -> List[ChatMessage]:
    """Get messages for a thread, creating a new thread if it doesn't exist"""
    if thread_id not in thread_memory:
        thread_memory[thread_id] = []
    return thread_memory[thread_id]

def update_thread_messages(thread_id: str, messages: List[ChatMessage]):
    """Update messages for a thread"""
    thread_memory[thread_id] = messages

def initialize_model():
    """Initialize the model and tokenizer"""
    global tokenizer, model, current_model_name
    try:
        current_model_name = DEFAULT_MODEL
        model_path = MODELS[current_model_name]
        logger.info(f"Loading model {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def switch_model(model_name: str):
    """Switch to a different model"""
    global tokenizer, model, current_model_name
    if model_name not in MODELS:
        raise ValueError("Unsupported model")
    
    try:
        logger.info(f"Switching to model {model_name}...")
        model_path = MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        current_model_name = model_name
        logger.info(f"Successfully switched to {model_name}")
    except Exception as e:
        error_msg = str(e)
        if "gemma" in model_name.lower() and ("gated repo" in error_msg or "restricted" in error_msg):
            raise ValueError("The Gemma model requires special access approval from Google. Please visit https://huggingface.co/google/gemma-2b-it to request access.")
        elif "gated repo" in error_msg or "restricted" in error_msg:
            raise ValueError(f"This model requires special access. Please visit {MODELS[model_name]} to request access.")
        logger.error(f"Error switching model: {e}")
        raise

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory Usage - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB")

def log_model_generation_params(model, **kwargs):
    """Log model generation parameters"""
    params = {
        "max_new_tokens": kwargs.get("max_new_tokens", 512),
        "temperature": kwargs.get("temperature", 0.0001),
        "do_sample": kwargs.get("do_sample", True),
        "top_p": kwargs.get("top_p", 0.5),
        "pad_token_id": kwargs.get("pad_token_id", model.config.pad_token_id),
        "use_cache": kwargs.get("use_cache", False),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
        "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 3)
    }
    logger.info(f"Model Generation Parameters: {json.dumps(params, indent=2)}")

def get_chat_response(messages: List[ChatMessage], model_name: str, thread_id: Optional[str] = None) -> ChatResponse:
    """Get response from a model for a given message and thread"""
    global tokenizer, model, current_model_name
    
    # Generate or use thread ID
    thread_id = thread_id or str(uuid.uuid4())
    
    # Get existing messages for this thread
    thread_messages = get_thread_messages(thread_id)
    
    # Combine existing messages with new messages
    all_messages = thread_messages + messages
    
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())
    logger.info(f"Starting chat request with ID: {request_id}")
    
    # Check if model needs to be changed
    if model_name != current_model_name:
        switch_model(model_name)
    
    try:
        start_time = time.time()
        logger.info(f"Starting chat request with {len(all_messages)} messages")
        # log_memory_usage()
        
        # Log complete conversation history
        #logger.info("Complete conversation history:")
        # for msg in all_messages:
        #     logger.info(f"{msg.role}: {msg.content}")
        
        # Prepare the prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in all_messages])
        prompt += "\nassistant:"
        
        # Log prompt details
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Full prompt content:\n{prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_tokens = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        
        # Log tokenization details
        logger.info(f"Input tokens count: {inputs.input_ids.shape[1]}")
        logger.info(f"Input tokens content (decoded):\n{input_tokens}")
        # logger.info(f"Input tokens IDs: {inputs.input_ids[0].tolist()}")
        
        # Log generation parameters
        generation_params = {
            "max_new_tokens": 512,
            "temperature": 0.0001,
            "do_sample": True,
            "top_p": 0.5,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "return_dict_in_generate": True  # Ensure we get a dictionary response
        }
        # log_model_generation_params(model, **generation_params)
        
        # Generate
        try:
            outputs = model.generate(
                **inputs,
                **generation_params
            )
            
            # Log generation metrics
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Handle different output formats
            if isinstance(outputs, dict) and 'sequences' in outputs:
                generated_tokens = outputs['sequences'].shape[1] - inputs.input_ids.shape[1]
                logger.info(f"Generated tokens: {generated_tokens}")
                response = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
            else:
                # Handle tensor output
                generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                logger.info(f"Generated tokens: {generated_tokens}")
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response (remove the prompt)
            response = response[len(prompt):].strip()
            
            # Log response details
            logger.info(f"Response length: {len(response)} characters")
            logger.info(f"Response content:\n{response}")
            # log_memory_usage()
            
            # Update thread memory with new messages and response
            new_messages = all_messages + [
                ChatMessage(role="assistant", content=response)
            ]
            update_thread_messages(thread_id, new_messages)
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return ChatResponse(
                response=response,
                thread_id=thread_id
            )
            
        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model generation error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in chat response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Log final metrics
        total_time = time.time() - start_time
        logger.info(f"Total request time: {total_time:.2f} seconds")
        # log_memory_usage()

def get_simple_response(message: str, model_name: str, thread_id: Optional[str] = None) -> ChatResponse:
    """Get a simple response from a model"""
    messages = [ChatMessage(role="user", content=message)]
    return get_chat_response(messages, model_name, thread_id)

# FastAPI endpoints
@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.get("/models")
async def get_models():
    """Get list of available models"""
    return {"models": list(MODELS.keys())}

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Get response from a model for a given message and thread"""
    try:
        return get_chat_response(request.messages, request.model, request.thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/response")
async def get_response(request: SimpleChatRequest) -> ChatResponse:
    """Simplified endpoint for getting a response from a model"""
    try:
        return get_simple_response(request.message, request.model, request.thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check the health of the service"""
    return {"status": "healthy", "model_loaded": model is not None}

def start_server(host: str = "127.0.0.1", port: int = 8000):
    """
    Start the FastAPI server with auto-reload enabled
    
    Args:
        host (str): Host address to bind to
        port (int): Port number to listen on
    """
    uvicorn.run(
        "src.backend.api:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload
        reload_dirs=["src/backend"],  # Watch these directories for changes
        reload_delay=1.0,  # Delay before reloading
        log_level="info"
    )

if __name__ == "__main__":
    start_server() 