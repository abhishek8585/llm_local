"""
Main script to run the LLM Chat Interface
This script can start both the frontend and API services.
"""

import subprocess
import sys
import os
from typing import Literal

def setup_environment():
    """Setup the Python environment for the application"""
    # Add the project root to PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    return env

def run_service(service: Literal["frontend", "api"]):
    env = setup_environment()
    
    if service == "frontend":
        subprocess.run(
            ["uv", "run", "streamlit", "run", "src/frontend/app.py"],
            env=env
        )
    elif service == "api":
        subprocess.run(
            ["uv", "run", "python", "src/backend/api.py"],
            env=env
        )

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["frontend", "api"]:
        print("Usage: python main.py [frontend|api]")
        sys.exit(1)
    
    run_service(sys.argv[1])
