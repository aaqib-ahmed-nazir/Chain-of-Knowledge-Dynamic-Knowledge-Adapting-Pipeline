from typing import Optional
from together import Together
import logging

logger = logging.getLogger(__name__)

class TogetherAIClient:
    """Together AI API client with caching and error handling."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3-70b-chat-hf", 
                 temperature: float = 0.0, max_tokens: int = 1024):
        """
        Initialize Together AI client.
        
        Args:
            api_key: Together AI API key
            model: Model to use (default: Llama-3-70b-chat)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.client = Together(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_cache = {}
        logger.info(f"TogetherAI client initialized with model: {model}")
    
    def call(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Call Together AI API with caching.
        
        Args:
            prompt: Input prompt
            temperature: Optional override for temperature
        
        Returns:
            Generated response
        """
        # Use cache
        cache_key = hash(prompt)
        if cache_key in self.request_cache:
            logger.debug("Cache hit")
            return self.request_cache[cache_key]
        
        temp = temperature if temperature is not None else self.temperature
        
        try:
            logger.debug(f"Calling Together AI (model={self.model}, temp={temp})")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content
            self.request_cache[cache_key] = result
            
            logger.debug(f"Together AI call successful ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Together AI API error: {str(e)}")
            raise

class LLMFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create_together_client(api_key: str, 
                               model: str = "meta-llama/Llama-3-70b-chat-hf",
                               temperature: float = 0.0) -> TogetherAIClient:
        """Create Together AI client."""
        return TogetherAIClient(api_key, model, temperature)
