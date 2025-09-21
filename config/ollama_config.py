"""
Ollama Configuration
Manages Ollama model paths and settings
"""

import os
from pathlib import Path
from typing import Optional

class OllamaConfig:
    """Configuration for Ollama models and paths"""
    
    def __init__(self):
        """Initialize Ollama configuration"""
        # Default model path (can be overridden)
        self.default_model_path = os.path.expanduser("~/.ollama/models")
        
        # Custom model path (set this to your desired folder)
        self.custom_model_path = os.path.expanduser("~/Documents/LLM_Models")
        
        # Ollama executable path
        self.ollama_executable = "ollama"
        
        # Default model name
        self.default_model = "llama3:8b-instruct"
        
        # Available models
        self.available_models = [
            "llama3:8b-instruct",
            "llama3:70b-instruct", 
            "llama2:7b",
            "llama2:13b",
            "mistral:7b-instruct",
            "codellama:7b-instruct",
            "phi3:3.8b"
        ]
    
    def get_model_path(self) -> str:
        """Get the configured model path"""
        # Check if custom path exists and is writable
        if os.path.exists(self.custom_model_path) and os.access(self.custom_model_path, os.W_OK):
            return self.custom_model_path
        
        # Fall back to default path
        return self.default_model_path
    
    def set_custom_model_path(self, path: str) -> bool:
        """
        Set custom model path
        
        Args:
            path: Path to store Ollama models
            
        Returns:
            bool: True if path is valid and accessible
        """
        try:
            expanded_path = os.path.expanduser(path)
            path_obj = Path(expanded_path)
            
            # Create directory if it doesn't exist
            path_obj.mkdir(parents=True, exist_ok=True)
            
            # Check if writable
            if os.access(expanded_path, os.W_OK):
                self.custom_model_path = expanded_path
                return True
            else:
                print(f"Warning: Path {expanded_path} is not writable")
                return False
                
        except Exception as e:
            print(f"Error setting custom model path: {str(e)}")
            return False
    
    def get_ollama_command(self, model: str = None) -> list:
        """
        Get Ollama command with proper environment
        
        Args:
            model: Model name to use
            
        Returns:
            list: Command to run Ollama
        """
        if model is None:
            model = self.default_model
        
        # Set environment variable for model path
        env = os.environ.copy()
        env['OLLAMA_MODELS'] = self.get_model_path()
        
        return [self.ollama_executable, "run", model]
    
    def get_ollama_serve_command(self) -> list:
        """
        Get Ollama serve command with custom model path
        
        Returns:
            list: Command to start Ollama serve
        """
        env = os.environ.copy()
        env['OLLAMA_MODELS'] = self.get_model_path()
        
        return [self.ollama_executable, "serve"]
    
    def check_model_exists(self, model: str) -> bool:
        """
        Check if a model exists in the configured path
        
        Args:
            model: Model name to check
            
        Returns:
            bool: True if model exists
        """
        try:
            import subprocess
            result = subprocess.run(
                [self.ollama_executable, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return model in result.stdout
            return False
            
        except Exception:
            return False
    
    def pull_model(self, model: str) -> bool:
        """
        Pull a model to the configured path
        
        Args:
            model: Model name to pull
            
        Returns:
            bool: True if successful
        """
        try:
            import subprocess
            
            # Set environment variable for model path
            env = os.environ.copy()
            env['OLLAMA_MODELS'] = self.get_model_path()
            
            result = subprocess.run(
                [self.ollama_executable, "pull", model],
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error pulling model {model}: {str(e)}")
            return False
    
    def get_available_models(self) -> list:
        """
        Get list of available models
        
        Returns:
            list: Available model names
        """
        try:
            import subprocess
            result = subprocess.run(
                [self.ollama_executable, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                models = []
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
            
        except Exception:
            return []

# Global configuration instance
ollama_config = OllamaConfig()
