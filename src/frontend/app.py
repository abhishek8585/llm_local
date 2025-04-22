"""
Streamlit UI for LLM Chat Interface
This file contains the main Streamlit application for the chat interface.
It allows users to interact with different LLMs through a web interface.
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import json
import time
import asyncio
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import API models after setting up the path
from src.backend.api import ChatMessage, ChatRequest, ChatResponse

class APIClient:
    """Client for interacting with the LLM API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def get_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()["models"]
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting models: {str(e)}")
            return ["phi-3-mini"]  # Fallback model
    
    def get_response(self, message: str, model: str, thread_id: Optional[str] = None) -> ChatResponse:
        """Get response from the model"""
        try:
            response = requests.post(
                f"{self.base_url}/response",
                json={
                    "model": model,
                    "message": message,
                    "thread_id": thread_id
                },
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            return ChatResponse(**response.json())
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. The model is taking longer than expected to respond. Please try again with a shorter prompt.")
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to the backend. Make sure it's running on http://localhost:8000")
        except Exception as e:
            raise Exception(f"Error getting response: {str(e)}")

def main():
    # Configure the page
    st.set_page_config(
        page_title="LLM Chat Interface",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    # Initialize API client
    api_client = APIClient()

    # Title and description
    st.title("🤖 LLM Chat Interface")
    st.markdown("""
        Chat with different LLMs using this interface. 
        Select a model and start chatting!
    """)

    # Sidebar for model selection
    with st.sidebar:
        st.header("Settings")
        # Get available models from API
        available_models = api_client.get_models()
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0
        )
        
        if st.button("New Chat"):
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show loading indicator
                with st.spinner("Thinking... This might take a few minutes for the first response."):
                    # Get response from API
                    response = api_client.get_response(
                        prompt,
                        selected_model,
                        st.session_state.thread_id
                    )
                    full_response = response.response
                    st.session_state.thread_id = response.thread_id
            except Exception as e:
                full_response = str(e)
            
            # Display the response
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    # Ensure we're in the main thread
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main() 