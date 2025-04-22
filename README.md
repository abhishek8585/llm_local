# LLM Chat Interface

A web interface for chatting with different Large Language Models (LLMs). Currently supports multiple models including Phi-3 Mini, Mistral-7B, DeepSeek models, and Gemma-2B (requires special access).

## Features

- Web-based chat interface using Streamlit
- FastAPI backend for model inference
- Support for chat history and thread management
- Multiple model support with easy switching
- Hugging Face integration for model access

## Prerequisites

- Python 3.12
- UV package manager
- CUDA-capable GPU (recommended)
- Hugging Face account and token (for accessing certain models)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-chat-ui
```

2. Install dependencies using UV:
```bash
uv add streamlit fastapi uvicorn transformers torch python-dotenv pydantic huggingface-hub
```

3. Create a `.env` file in the project root and add your Hugging Face token:
```
HUGGINGFACE_TOKEN=your_token_here
```

## Running the Application

The application consists of two services that need to be run separately:

1. Start the backend service:
```bash
python main.py backend
```

2. In a separate terminal, start the frontend service:
```bash
python main.py frontend
```

The frontend will be available at `http://localhost:8501` and the backend API at `http://localhost:8000`.

## Project Structure

```
.
├── src/
│   ├── frontend/
│   │   └── app.py          # Streamlit UI
│   ├── backend/
│   │   └── main.py         # FastAPI backend
│   └── models/
│       └── ...             # Model-specific code
├── data/                   # Data directory
├── main.py                 # Service launcher
├── .env                    # Environment variables
└── pyproject.toml          # Project dependencies
```

## API Documentation

### Chat Endpoint

**Endpoint:** `POST /chat`

Send chat messages and receive responses from the selected model.

#### Request Body

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you!"
        }
    ],
    "model": "phi-3-mini",
    "thread_id": "optional_thread_id"
}
```

##### Parameters:

- `messages` (array): List of chat messages
  - Each message has:
    - `role` (string): Either "user" or "assistant"
    - `content` (string): The message content
- `model` (string): The model to use for generation
  - Available models:
    - `phi-3-mini` (default)
    - `mistral-7b`
    - `deepseek-coder-7b`
    - `deepseek-v2.5`
    - `gemma-2b` (requires special access)
- `thread_id` (string, optional): Identifier for chat thread/context

#### Response

```json
{
    "response": "I'm an AI assistant, how can I help you today?",
    "thread_id": "thread_123"
}
```

##### Response Fields:

- `response` (string): The model's generated response
- `thread_id` (string): Identifier for the chat thread

#### Example Usage

```python
import requests

# Example chat request
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "model": "phi-3-mini"
    }
)

print(response.json())
```

### Health Check Endpoint

**Endpoint:** `GET /health`

Check the service health and model status.

#### Response

```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## Model Access Notes

- Most models are publicly available
- The Gemma model requires special access from Google:
  1. Visit https://huggingface.co/google/gemma-2b-it
  2. Click "Access repository"
  3. Accept Google's terms of use
  4. Wait for approval

## License

[Add your license here]
