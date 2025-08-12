"""
Web Scraping Modules for Business Contact Research
"""

# Make modules importable
from . import web_scraping_module
from . import streamlit_business_researcher
from . import enhanced_government_researcher

__all__ = [
    'web_scraping_module',
    'streamlit_business_researcher', 
    'enhanced_government_researcher'
]

try:
    from .enhanced_timber_business_researcher import EnhancedTimberBusinessResearcher, research_timber_businesses_from_dataframe
except ImportError:
    # Module may not be available in all deployments
    pass
