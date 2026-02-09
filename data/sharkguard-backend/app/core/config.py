import os

class Config:
    """Configuration settings for the application."""
    
    # General settings
    APP_NAME = os.getenv("APP_NAME", "SharkGuard")
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # API settings
    API_V1_STR = "/api/v1"
    
    # Database settings (if applicable)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # Etherscan settings
    ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
    
    # Other settings can be added here as needed
