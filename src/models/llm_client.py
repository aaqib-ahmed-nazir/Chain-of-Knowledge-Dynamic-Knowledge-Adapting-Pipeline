import os
import time
from typing import Optional, List
from groq import Groq
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_cache = {}
    
    def call(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call Groq API with caching and rate limit handling."""
        cache_key = hash(prompt)
        if cache_key in self.request_cache:
            logger.debug("Cache hit")
            return self.request_cache[cache_key]
        
        temp = temperature if temperature is not None else self.temperature
        
        max_retries = 3
        base_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                message = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=self.max_tokens
                )
                response = message.choices[0].message.content
                self.request_cache[cache_key] = response
                logger.debug(f"API call successful (model={self.model}, temp={temp})")
                return response
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error
                if 'rate_limit' in error_str.lower() or '429' in error_str:
                    if attempt < max_retries - 1:
                        # Extract wait time from error if available
                        wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                        if 'try again in' in error_str:
                            try:
                                # Try to extract wait time from error message (handles both formats)
                                import re
                                # Try format: "try again in 7m37.056s" or "try again in 1m22.08s"
                                match = re.search(r'try again in (?:(\d+)m)?(?:(\d+\.?\d*)s)?', error_str)
                                if match:
                                    minutes = int(match.group(1) or 0)
                                    seconds = float(match.group(2) or 0)
                                    wait_time = (minutes * 60) + seconds + 10  # Add 10s buffer
                                    logger.warning(f"Extracted wait time from error: {wait_time:.1f}s")
                            except Exception as parse_error:
                                logger.debug(f"Could not parse wait time: {parse_error}")
                                pass
                        
                        logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries")
                        # Extract wait time for final error message
                        wait_time_msg = ""
                        if 'try again in' in error_str:
                            try:
                                import re
                                match = re.search(r'try again in (?:(\d+)m)?(?:(\d+\.?\d*)s)?', error_str)
                                if match:
                                    minutes = int(match.group(1) or 0)
                                    seconds = float(match.group(2) or 0)
                                    total_seconds = (minutes * 60) + seconds
                                    wait_time_msg = f" Please wait {total_seconds/60:.1f} minutes before retrying."
                            except:
                                pass
                        raise Exception(f"Rate limit exceeded after {max_retries} retries.{wait_time_msg}")
                else:
                    logger.error(f"Groq API error: {error_str}")
                    raise
        
        raise Exception("Failed to get response after retries")

class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_cache = {}
    
    def call(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call Gemini API with caching."""
        cache_key = hash(prompt)
        if cache_key in self.request_cache:
            logger.debug("Cache hit")
            return self.request_cache[cache_key]
        
        temp = temperature if temperature is not None else self.temperature
        
        try:
            model = genai.GenerativeModel(self.model_name)
            config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=self.max_tokens
            )
            response = model.generate_content(prompt, generation_config=config)
            
            # Handle blocked content (finish_reason == 2)
            if response.candidates and response.candidates[0].finish_reason == 2:
                logger.warning("Content was blocked by safety filters - trying alternative prompt")
                # Try with a more neutral prompt
                try:
                    safe_prompt = f"Please provide factual information about: {prompt[:200]}"
                    response = model.generate_content(safe_prompt, generation_config=config)
                    if response.candidates and response.candidates[0].finish_reason != 2:
                        result = response.text
                        self.request_cache[cache_key] = result
                        return result
                except:
                    pass
                logger.warning("Content was blocked by safety filters")
                return "I cannot provide a response due to content safety filters."
            
            result = response.text
            self.request_cache[cache_key] = result
            logger.debug(f"API call successful (model={self.model_name}, temp={temp})")
            return result
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Return a fallback response instead of raising
            return f"Error generating response: {str(e)}"

class LLMFactory:
    @staticmethod
    def create_groq_client(api_key: str, model: str = "llama-3.3-70b-versatile", 
                          temperature: float = 0.0) -> GroqClient:
        return GroqClient(api_key, model, temperature)
    
    @staticmethod
    def create_gemini_client(api_key: str, model: str = "gemini-2.5-flash",
                            temperature: float = 0.0) -> GeminiClient:
        return GeminiClient(api_key, model, temperature)

