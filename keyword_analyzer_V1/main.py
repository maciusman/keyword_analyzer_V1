"""
Główny skrypt do uruchomienia aplikacji Analizatora Słów Kluczowych.
"""
import streamlit.web.cli as stcli
import sys
import os
import logging
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("keyword_analyzer_V1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_api_keys():
    """
    Sprawdza, czy klucze API są ustawione.
    
    Returns:
        bool: True, jeśli wszystkie klucze są ustawione, False w przeciwnym razie
    """
    # Załaduj zmienne środowiskowe
    load_dotenv()
    
    # Sprawdź klucze API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    jina_api_key = os.getenv("JINA_API_KEY")
    
    is_valid = True
    
    if not gemini_api_key:
        logger.error("Brak klucza API dla Gemini. Ustaw go w pliku .env")
        is_valid = False
    
    if not jina_api_key:
        logger.error("Brak klucza API dla JINA AI. Ustaw go w pliku .env")
        is_valid = False
    
    return is_valid

def main():
    """
    Funkcja główna uruchamiająca aplikację.
    """
    logger.info("Uruchamianie Analizatora Słów Kluczowych...")
    
    # Sprawdź, czy klucze API są ustawione
    if not check_api_keys():
        logger.error("Brak wymaganych kluczy API. Zatrzymuję aplikację.")
        sys.exit(1)
    
    # Sprawdź, czy istnieją wymagane katalogi
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # Uruchom aplikację Streamlit
    logger.info("Uruchamianie interfejsu Streamlit...")
    
    # Przygotuj argumenty dla Streamlit CLI
    sys.argv = ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=localhost"]
    
    # Uruchom Streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
