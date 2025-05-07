"""
Moduł do generowania embedingów słów kluczowych za pomocą JINA AI.
Implementacja używająca bezpośrednio API REST zamiast SDK.
"""
import logging
import requests
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import time

# Import konfiguracji
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import JINA_API_KEY, JINA_EMBEDDING_MODEL

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Klasa odpowiedzialna za generowanie embedingów dla słów kluczowych.
    Używa bezpośrednio JINA AI REST API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Inicjalizuje generator embedingów.
        
        Args:
            api_key: Klucz API dla JINA AI (opcjonalnie, domyślnie z config)
            model_name: Nazwa modelu JINA AI (opcjonalnie, domyślnie z config)
        """
        self.api_key = api_key or JINA_API_KEY
        self.model_name = model_name or JINA_EMBEDDING_MODEL
        
        # Sprawdzenie, czy klucz API jest dostępny
        if not self.api_key:
            raise ValueError("Brak klucza API dla JINA AI. Ustaw go w pliku .env.")
        
        # Adres API JINA AI
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        logger.info(f"Zainicjalizowano klienta JINA AI REST z modelem {self.model_name}")
    
    def generate_embeddings(self, keywords: List[str], batch_size: int = 20, 
                            progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """
        Generuje embedingi dla listy słów kluczowych.
        
        Args:
            keywords: Lista słów kluczowych
            batch_size: Rozmiar partii do przetwarzania (unikanie limitów API)
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Tablica numpy zawierająca embedingi
        """
        logger.info(f"Generowanie embedingów dla {len(keywords)} słów kluczowych...")
        
        embeddings = []
        batches = [keywords[i:i+batch_size] for i in range(0, len(keywords), batch_size)]
        total_batches = len(batches)
        
        # Wybór między tqdm a progress_callback
        if progress_callback:
            iterator = enumerate(batches)
        else:
            iterator = enumerate(tqdm(batches, desc="Generowanie embedingów"))
        
        for batch_idx, batch in iterator:
            try:
                # Przygotuj dane dla API
                input_data = [{"text": keyword} for keyword in batch]
                
                # Przygotuj payload
                payload = {
                    "model": self.model_name,
                    "input": input_data
                }
                
                # Wykonaj zapytanie do API
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload
                )
                
                # Sprawdź, czy zapytanie było udane
                if response.status_code == 200:
                    # Pobierz embedingi z odpowiedzi
                    response_data = response.json()
                    batch_embeddings = response_data.get('data', [])
                    
                    # Pobierz rzeczywiste embedingi
                    batch_vectors = []
                    for item in batch_embeddings:
                        embedding = item.get('embedding', [])
                        batch_vectors.append(embedding)
                    
                    # Dodaj embedingi do wyników
                    embeddings.extend(batch_vectors)
                else:
                    logger.error(f"Błąd API: {response.status_code}, {response.text}")
                    # W przypadku błędu API, dodajemy puste embeddiny
                    for _ in batch:
                        embeddings.append(np.zeros(512))  # Typowy rozmiar
                
                # Aktualizuj pasek postępu
                if progress_callback:
                    progress = (batch_idx + 1) / total_batches
                    status = f"Generowanie embedingów: partia {batch_idx + 1}/{total_batches}"
                    progress_callback(progress, status)
                
                # Krótka przerwa, aby uniknąć limitów API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Błąd podczas generowania embedingów dla partii: {e}")
                # W przypadku błędu, dodajemy puste embeddiny (zeros) dla tej partii
                for _ in batch:
                    embeddings.append(np.zeros(512))  # Typowy rozmiar embedingu
        
        # Konwersja listy embedingów na tablicę numpy
        embeddings_array = np.array(embeddings)
        
        logger.info(f"Wygenerowano embedingi o wymiarach: {embeddings_array.shape}")
        return embeddings_array
    
    def process_keywords_dataframe(self, df: pd.DataFrame, keyword_col: str = 'keyword',
                                   progress_callback: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
        """
        Przetwarza DataFrame z słowami kluczowymi, dodając embedingi.
        
        Args:
            df: DataFrame zawierający słowa kluczowe
            keyword_col: Nazwa kolumny zawierającej słowa kluczowe
            progress_callback: Funkcja callback dla aktualizacji paska postępu
            
        Returns:
            DataFrame z dodanymi embedingami jako nową kolumną 'embedding'
        """
        if keyword_col not in df.columns:
            raise ValueError(f"Kolumna {keyword_col} nie istnieje w DataFrame")
        
        # Pobierz listę słów kluczowych
        keywords_list = df[keyword_col].tolist()
        
        # Generuj embedingi
        embeddings = self.generate_embeddings(keywords_list, progress_callback=progress_callback)
        
        # Stwórz kopię DataFrame, aby nie modyfikować oryginału
        result_df = df.copy()
        
        # Dodaj embedingi jako nową kolumnę
        result_df['embedding'] = list(embeddings)
        
        return result_df