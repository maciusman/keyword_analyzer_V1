"""
Moduł do klastrowania słów kluczowych na podstawie ich embedingów.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
import umap
from tqdm import tqdm

# Import konfiguracji
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CLUSTERING_METHOD, UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, 
    UMAP_MIN_DIST, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
)

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordClusterer:
    """
    Klasa odpowiedzialna za klastrowanie słów kluczowych.
    """
    
    def __init__(self, 
                 method: Optional[str] = None,
                 umap_n_neighbors: Optional[int] = None,
                 umap_n_components: Optional[int] = None,
                 umap_min_dist: Optional[float] = None,
                 hdbscan_min_cluster_size: Optional[int] = None,
                 hdbscan_min_samples: Optional[int] = None):
        """
        Inicjalizuje klasterer słów kluczowych.
        
        Args:
            method: Metoda klastrowania ('hdbscan', 'dbscan', lub 'kmeans')
            umap_n_neighbors: Parametr n_neighbors dla UMAP
            umap_n_components: Liczba wymiarów po redukcji UMAP
            umap_min_dist: Parametr min_dist dla UMAP
            hdbscan_min_cluster_size: Minimalny rozmiar klastra dla HDBSCAN
            hdbscan_min_samples: Minimalny rozmiar próbki dla HDBSCAN
        """
        self.method = method or CLUSTERING_METHOD
        self.umap_n_neighbors = umap_n_neighbors or UMAP_N_NEIGHBORS
        self.umap_n_components = umap_n_components or UMAP_N_COMPONENTS
        self.umap_min_dist = umap_min_dist or UMAP_MIN_DIST
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size or HDBSCAN_MIN_CLUSTER_SIZE
        self.hdbscan_min_samples = hdbscan_min_samples or HDBSCAN_MIN_SAMPLES
        
        self.umap_model = None
        self.cluster_model = None
        self.reduced_embeddings = None
        
        logger.info(f"Zainicjalizowano KeywordClusterer z metodą: {self.method}")
    
    def reduce_dimensions(self, embeddings: np.ndarray, 
                          progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """
        Redukuje wymiarowość embedingów za pomocą UMAP.
        
        Args:
            embeddings: Tablica numpy zawierająca embedingi
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Tablica numpy zawierająca zredukowane embedingi
        """
        logger.info(f"Redukowanie wymiarowości z {embeddings.shape[1]} do {self.umap_n_components} wymiarów...")
        
        if progress_callback:
            progress_callback(0.1, "Inicjalizacja UMAP...")
        
        # Inicjalizacja modelu UMAP
        self.umap_model = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=self.umap_n_components,
            min_dist=self.umap_min_dist,
            random_state=42
        )
        
        if progress_callback:
            progress_callback(0.3, "Reducja wymiarowości przy użyciu UMAP...")
        
        # Dopasowanie i transformacja danych
        reduced_embeddings = self.umap_model.fit_transform(embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Redukcja wymiarowości zakończona")
        
        logger.info(f"Zredukowano wymiarowość do: {reduced_embeddings.shape}")
        self.reduced_embeddings = reduced_embeddings
        return reduced_embeddings
    
    def cluster_keywords(self, embeddings: np.ndarray, n_clusters: Optional[int] = None,
                         progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """
        Klastruje słowa kluczowe na podstawie ich embedingów.
        
        Args:
            embeddings: Tablica numpy zawierająca embedingi
            n_clusters: Liczba klastrów (tylko dla metody 'kmeans')
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Tablica numpy zawierająca etykiety klastrów
        """
        # Redukcja wymiarowości przed klastrowaniem
        if self.reduced_embeddings is None or len(self.reduced_embeddings) != len(embeddings):
            reduced_data = self.reduce_dimensions(embeddings, progress_callback)
        else:
            reduced_data = self.reduced_embeddings
            if progress_callback:
                progress_callback(0.5, "Używanie zredukowanych embedingów...")
        
        # Klastrowanie
        if self.method.lower() == 'hdbscan':
            if progress_callback:
                progress_callback(0.6, "Inicjalizacja HDBSCAN...")
            
            logger.info(f"Klastrowanie za pomocą HDBSCAN (min_cluster_size={self.hdbscan_min_cluster_size}, min_samples={self.hdbscan_min_samples})...")
            self.cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                gen_min_span_tree=True,
                core_dist_n_jobs=-1
            )
            
            if progress_callback:
                progress_callback(0.7, "Klastrowanie HDBSCAN w toku...")
            
            labels = self.cluster_model.fit_predict(reduced_data)
            
        elif self.method.lower() == 'dbscan':
            if progress_callback:
                progress_callback(0.6, "Inicjalizacja DBSCAN...")
            
            logger.info("Klastrowanie za pomocą DBSCAN...")
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
            
            if progress_callback:
                progress_callback(0.7, "Klastrowanie DBSCAN w toku...")
            
            labels = self.cluster_model.fit_predict(reduced_data)
            
        elif self.method.lower() == 'kmeans':
            # Jeśli nie podano liczby klastrów, użyj heurystyki
            if n_clusters is None:
                n_clusters = min(int(np.sqrt(len(reduced_data))), 100)
                logger.info(f"Nie podano liczby klastrów. Używam n_clusters={n_clusters}")
            
            if progress_callback:
                progress_callback(0.6, f"Inicjalizacja KMeans ({n_clusters} klastrów)...")
            
            logger.info(f"Klastrowanie za pomocą KMeans (n_clusters={n_clusters})...")
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            
            if progress_callback:
                progress_callback(0.7, "Klastrowanie KMeans w toku...")
            
            labels = self.cluster_model.fit_predict(reduced_data)
            
        else:
            raise ValueError(f"Nieznana metoda klastrowania: {self.method}")
        
        # Liczba unikalnych klastrów (bez szumu, który ma etykietę -1)
        unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Zakończono klastrowanie. Liczba klastrów: {unique_clusters}")
        
        if progress_callback:
            progress_callback(1.0, f"Klastrowanie zakończone - {unique_clusters} klastrów")
        
        return labels
    
    def process_keywords_dataframe(self, df: pd.DataFrame, embedding_col: str = 'embedding', 
                                   n_clusters: Optional[int] = None,
                                   progress_callback: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
        """
        Przetwarza DataFrame z embedingami, dodając etykiety klastrów.
        
        Args:
            df: DataFrame zawierający embedingi
            embedding_col: Nazwa kolumny zawierającej embedingi
            n_clusters: Liczba klastrów (tylko dla metody 'kmeans')
            progress_callback: Funkcja callback dla aktualizacji paska postępu
            
        Returns:
            DataFrame z dodanymi etykietami klastrów jako nową kolumną 'cluster'
        """
        if embedding_col not in df.columns:
            raise ValueError(f"Kolumna {embedding_col} nie istnieje w DataFrame")
        
        # Pobierz embedingi
        embeddings = np.array(df[embedding_col].tolist())
        
        # Klastruj słowa kluczowe
        labels = self.cluster_keywords(embeddings, n_clusters, progress_callback)
        
        # Stwórz kopię DataFrame, aby nie modyfikować oryginału
        result_df = df.copy()
        
        # Dodaj etykiety klastrów jako nową kolumnę
        result_df['cluster'] = labels
        
        # Sortuj DataFrame według klastrów
        result_df = result_df.sort_values(by='cluster')
        
        return result_df
    
    def get_cluster_keywords(self, df: pd.DataFrame, cluster_id: int, keyword_col: str = 'keyword') -> List[str]:
        """
        Pobiera słowa kluczowe należące do danego klastra.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            cluster_id: ID klastra
            keyword_col: Nazwa kolumny zawierającej słowa kluczowe
            
        Returns:
            Lista słów kluczowych należących do klastra
        """
        if 'cluster' not in df.columns:
            raise ValueError("Kolumna 'cluster' nie istnieje w DataFrame. Najpierw wykonaj klastrowanie.")
        
        # Filtruj DataFrame według ID klastra
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Pobierz słowa kluczowe
        keywords = cluster_df[keyword_col].tolist()
        
        return keywords
    
    def get_cluster_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generuje statystyki dla każdego klastra.
        
        Args:
            df: DataFrame zawierający słowa kluczowe, etykiety klastrów i metryki
            
        Returns:
            DataFrame ze statystykami klastrów
        """
        if 'cluster' not in df.columns:
            raise ValueError("Kolumna 'cluster' nie istnieje w DataFrame. Najpierw wykonaj klastrowanie.")
        
        # Grupuj według klastrów i oblicz statystyki
        stats = df.groupby('cluster').agg({
            'keyword': 'count',
            'volume': 'sum',
            'difficulty': 'mean',
            'cpc': 'mean'
        }).reset_index()
        
        # Zmień nazwy kolumn
        stats = stats.rename(columns={
            'keyword': 'count',
            'volume': 'total_volume',
            'difficulty': 'avg_difficulty',
            'cpc': 'avg_cpc'
        })
        
        # Sortuj według liczby słów kluczowych (malejąco)
        stats = stats.sort_values(by='count', ascending=False)
        
        return stats