"""
Plik konfiguracyjny zawierający ustawienia API i parametry analizy.
"""
import os
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych z pliku .env
load_dotenv()

# Klucze API (bezpieczniej jest przechowywać je w zmiennych środowiskowych)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Opis dostępnych modeli Gemini
# 
# Wariant modelu: Gemini 2.5 Flash Preview 04-17 (gemini-2.5-flash-preview-04-17)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst
# Zoptymalizowany dla: Elastyczne myślenie, opłacalność
#
# Wariant modelu: Podgląd Gemini 2.5 Pro (gemini-2.5-pro-preview-05-06)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst
# Zoptymalizowany dla: Ulepszone myślenie i rozumowanie, zrozumienie multimodalne, zaawansowane kodowanie i inne funkcje
#
# Wariant modelu: Gemini 2.0 Flash (gemini-2.0-flash)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst, obrazy (w wersji eksperymentalnej) i dźwięk (wkrótce)
# Zoptymalizowany dla: Funkcje nowej generacji, szybkość, myślenie, strumieniowanie w czasie rzeczywistym i generowanie multimodalne
#
# Wariant modelu: Gemini 2.0 Flash-Lite (gemini-2.0-flash-lite)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst
# Zoptymalizowany dla: Opłacalność i niskie opóźnienie
#
# Wariant modelu: Gemini 1.5 Flash (gemini-1.5-flash)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst
# Zoptymalizowany dla: Szybkie i wszechstronne działanie w różnych zastosowaniach
#
# Wariant modelu: Gemini 1.5 Flash-8B (gemini-1.5-flash-8b)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst
# Zoptymalizowany dla: Zadania o dużej liczbie i mniejszym zaawansowaniu
#
# Wariant modelu: Gemini 1.5 Pro (gemini-1.5-pro)
# Dane wejściowe: Dźwięk, obrazy, filmy i tekst
# Wyniki: Tekst
# Zoptymalizowany dla: Złożone zadania wymagające większej inteligencj


# Konfiguracja modeli
GEMINI_MODEL = "gemini-2.0-flash"  # Nazwa modelu Gemini
JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en"  # Model JINA AI do embedingów

# Parametry analizy
CLUSTERING_METHOD = "hdbscan"  # Metoda klastrowania (hdbscan, dbscan, kmeans)
UMAP_N_NEIGHBORS = 15  # Parametr n_neighbors dla UMAP
# UWAGA: Rozważ zwiększenie UMAP_N_COMPONENTS dla lepszego klastrowania (np. 10-30),
# a osobną redukcję do 2/3 tylko dla wizualizacji.
UMAP_N_COMPONENTS = 10  # Liczba wymiarów po redukcji UMAP
UMAP_MIN_DIST = 0.2  # Parametr min_dist dla UMAP
# UWAGA: Rozważ eksperymenty z mniejszym HDBSCAN_MIN_CLUSTER_SIZE (np. 5, 8, 10)
HDBSCAN_MIN_CLUSTER_SIZE = 15  # Minimalny rozmiar klastra dla HDBSCAN
HDBSCAN_MIN_SAMPLES = 5  # Minimalny rozmiar próbki dla HDBSCAN

# Parametry priorytetyzacji (używane w analyzer.py)
VOLUME_WEIGHT = 0.4      # Waga wolumenu wyszukiwań
KD_WEIGHT = 0.3          # Waga trudności słowa kluczowego
CPC_WEIGHT = 0.3         # Waga kosztu kliknięcia

# Parametry normalizacji dla _calculate_priority_score
# Określają przybliżoną wartość, przy której metryka osiąga maksymalny wpływ (score=1.0)
MAX_EXPECTED_VOLUME_FOR_SCALING = 50000 # Wolumen, powyżej którego score = 1.0 w normalizacji
MAX_EXPECTED_CPC_FOR_SCALING = 10.0      # CPC, powyżej którego score = 1.0 w normalizacji

# Wpływ oceny modelu Gemini na końcowy priorytet
# Wyższa wartość -> większy wpływ oceny modelu (High/Medium/Low)
MODEL_PRIORITY_WEIGHT = 1.5

# Progi dla końcowych poziomów priorytetów (Niski/Średni/Wysoki)
# Dostosuj te wartości po testach, aby uzyskać pożądany rozkład!
# Bazują na combined_priority obliczonym w analyzer.py
# Typowy zakres combined_priority po zmianach to ~[0, 1.5], ale może być szerszy
LOW_PRIORITY_THRESHOLD = 0.5    # Wynik < tej wartości -> Niski
MEDIUM_PRIORITY_THRESHOLD = 0.8   # Wynik >= tej wartości -> Wysoki (inaczej Średni)


# Parametry UMAP dla wizualizacji 3D (używane w ui/app.py)
UMAP_VIS_N_NEIGHBORS = 15  # Możesz dostosować, często podobne do UMAP_N_NEIGHBORS
UMAP_VIS_MIN_DIST = 0.1    # Możesz dostosować, często niższe dla lepszego rozdzielenia wizualnego
# UMAP_VIS_N_COMPONENTS będzie na sztywno 3 w kodzie ui/app.py

# Ścieżki do plików
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Upewnij się, że katalogi istnieją
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Sprawdzenie kluczy API (Opcjonalne, ale pomocne) ---
# if not GEMINI_API_KEY:
#     print("OSTRZEŻENIE: Brak klucza GEMINI_API_KEY w .env lub zmiennych środowiskowych.")
# if not JINA_API_KEY and JINA_EMBEDDING_MODEL.startswith("jina-"):
#     print("OSTRZEŻENIE: Brak klucza JINA_API_KEY, potrzebny dla modeli JINA.")

# <<<< NOWA SEKCJA DODANA NA KOŃCU >>>>
# --- Konfiguracja OpenRouter ---
# Bazowy URL dla API OpenRouter v1
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1" 
# Endpoint do pobierania listy modeli
OPENROUTER_MODELS_ENDPOINT = "/models" 
# Endpoint do zapytań typu chat completion
OPENROUTER_CHAT_ENDPOINT = "/chat/completions" 

# Klucz API OpenRouter - Ładowany z .env, domyślnie pusty
# Użytkownik będzie mógł go nadpisać w UI
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") 
# <<<< KONIEC NOWEJ SEKCJI >>>>


# --- OpenRouter API ---
# Klucz API dla OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Konfiguracja endpointów OpenRouter
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = "/chat/completions"
OPENROUTER_MODELS_ENDPOINT = "/models"

# Można dodać domyślny model OpenRouter, jeśli chcesz
# OPENROUTER_DEFAULT_MODEL = "anthropic/claude-3-opus-20240229"