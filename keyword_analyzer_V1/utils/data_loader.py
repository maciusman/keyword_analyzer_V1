"""
Moduł odpowiedzialny za wczytywanie i wstępne przetwarzanie danych z Ahrefs.
"""
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
# Dodatkowe importy dla zapisu/odczytu analizy
import json
import pickle
import numpy as np
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ahrefs_export(file_path: str) -> pd.DataFrame:
    """
    Wczytuje plik eksportu z Ahrefs (CSV lub Excel) i zwraca DataFrame.
    
    Args:
        file_path: Ścieżka do pliku eksportu z Ahrefs
        
    Returns:
        DataFrame zawierający dane z Ahrefs
    """
    logger.info(f"Wczytywanie pliku: {file_path}")
    
    # Określenie rozszerzenia pliku
    _, file_extension = os.path.splitext(file_path)
    
    try:
        # Wczytywanie w zależności od formatu
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {file_extension}")
        
        logger.info(f"Pomyślnie wczytano dane. Liczba wierszy: {len(df)}")
        return df
    
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania pliku: {e}")
        raise

def preprocess_ahrefs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przetwarza wstępnie dane z Ahrefs, mapując nazwy kolumn i czyszcząc dane.
    
    Args:
        df: DataFrame z danymi z Ahrefs
        
    Returns:
        Przetworzony DataFrame
    """
    logger.info("Rozpoczynam wstępne przetwarzanie danych...")
    
    # Przekształcamy nazwy kolumn na małe litery
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Mapowanie nazw kolumn (różne wersje eksportu mogą mieć różne nazwy)
    column_mapping = {
        'keyword': 'keyword',
        'słowo kluczowe': 'keyword',
        'volume': 'volume',
        'wolumen': 'volume',
        'monthly volume': 'volume',
        'difficulty': 'difficulty',
        'kd': 'difficulty',
        'keyword difficulty': 'difficulty',
        'trudność': 'difficulty',
        'cpc': 'cpc',
        'cost per click': 'cpc',
        'koszt kliknięcia': 'cpc',
        'search intent': 'intent',
        'intencja wyszukiwania': 'intent',
        'position': 'position',
        'pozycja': 'position',
        'branded': 'branded',
        'local': 'local',
        'navigational': 'navigational',
        'informational': 'informational',
        'commercial': 'commercial',
        'transactional': 'transactional'
    }
    
    # Standardyzujemy nazwy kolumn
    renamed_columns = {}
    for col in df.columns:
        for original, standard in column_mapping.items():
            if col == original or col.startswith(original):
                renamed_columns[col] = standard
                break
    
    # Zmieniamy nazwy kolumn (tylko te, które zostały zmapowane)
    df = df.rename(columns=renamed_columns)
    
    # Upewniamy się, że mamy przynajmniej kolumnę z frazami kluczowymi
    if 'keyword' not in df.columns:
        raise ValueError("Brak kolumny ze słowami kluczowymi w danych")
    
    # Obsługa brakujących kolumn
    if 'volume' not in df.columns:
        logger.warning("Brak kolumny z wolumenem wyszukiwań. Dodaję kolumnę z wartością domyślną 0.")
        df['volume'] = 0
    else:
        # Konwersja wolumenu na liczby
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    
    if 'difficulty' not in df.columns:
        logger.warning("Brak kolumny z trudnością słów kluczowych. Dodaję kolumnę z wartością domyślną 0.")
        df['difficulty'] = 0
    else:
        # Konwersja trudności na liczby
        df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').fillna(0)
    
    if 'cpc' not in df.columns:
        logger.warning("Brak kolumny z CPC. Dodaję kolumnę z wartością domyślną 0.")
        df['cpc'] = 0.0
    else:
        # Konwersja CPC na liczby
        df['cpc'] = pd.to_numeric(df['cpc'], errors='coerce').fillna(0.0)
    
    if 'intent' not in df.columns:
        logger.warning("Brak kolumny z intencją wyszukiwania. Dodaję kolumnę z wartością 'unknown'.")
        df['intent'] = 'unknown'
    
    # Czyszczenie danych
    logger.info("Czyszczenie danych...")
    
    # Usuwamy duplikaty słów kluczowych
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['keyword'])
    after_dedup = len(df)
    if before_dedup > after_dedup:
        logger.info(f"Usunięto {before_dedup - after_dedup} duplikatów.")
    
    # Usuwamy wiersze z pustymi słowami kluczowymi
    df = df[df['keyword'].notna() & (df['keyword'] != '')]
    
    # Standaryzacja wartości w kolumnie intent
    if 'intent' in df.columns:
        # Mapowanie różnych formatów intencji na standardowe wartości
        intent_mapping = {
            'informational': 'informational',
            'informacyjna': 'informational',
            'commercial': 'commercial',
            'komercyjna': 'commercial',
            'transactional': 'transactional',
            'transakcyjna': 'transactional',
            'navigational': 'navigational',
            'nawigacyjna': 'navigational'
        }
        
        # Standaryzacja wartości intencji
        df['intent'] = df['intent'].str.lower().map(intent_mapping).fillna('unknown')
    
    # Obsługa kolumn intencji z Ahrefs (Branded, Local, Navigational, Informational, Commercial, Transactional)
    intent_columns = ['branded', 'local', 'navigational', 'informational', 'commercial', 'transactional']
    available_intent_columns = [col for col in intent_columns if col in df.columns]
    
    if available_intent_columns:
        logger.info(f"Wykryto kolumny intencji: {available_intent_columns}")
        
        # Konwersja wartości na typ boolean
        for col in available_intent_columns:
            if df[col].dtype == 'object':
                # Konwersja wartości tekstowych 'True'/'False' na boolean
                df[col] = df[col].map({'True': True, 'False': False, 
                                        'true': True, 'false': False,
                                        True: True, False: False}).fillna(False)
            else:
                # Upewniamy się, że kolumna jest typu boolean
                df[col] = df[col].astype(bool)
        
        # Utworzenie listy intencji dla każdego słowa kluczowego
        df['intent_list'] = df.apply(
            lambda row: [intent for intent in available_intent_columns if row[intent] == True],
            axis=1
        )
        
        # Jeśli słowo nie ma przypisanej żadnej intencji, oznacza jako 'unknown'
        df['intent_list'] = df['intent_list'].apply(lambda x: x if x else ['unknown'])
    else:
        logger.warning("Nie wykryto kolumn intencji z Ahrefs. Używam domyślnej kolumny 'intent'.")
        # Tworzenie intent_list na podstawie kolumny intent
        df['intent_list'] = df['intent'].apply(lambda x: [x] if x != 'unknown' else ['unknown'])
    
    logger.info(f"Zakończono wstępne przetwarzanie. Liczba wierszy po przetworzeniu: {len(df)}")
    return df

def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generuje podstawowe statystyki dla danych.
    
    Args:
        df: DataFrame z danymi
        
    Returns:
        Słownik ze statystykami
    """
    stats = {
        'total_keywords': len(df),
        'total_volume': int(df['volume'].sum()),
        'avg_difficulty': round(df['difficulty'].mean(), 2),
        'avg_cpc': round(df['cpc'].mean(), 2),
    }
    
    # Analiza intencji
    intent_columns = ['branded', 'local', 'navigational', 'informational', 'commercial', 'transactional']
    available_intent_columns = [col for col in intent_columns if col in df.columns]
    
    if available_intent_columns:
        # Obliczanie statystyk dla każdej intencji z kolumn Ahrefs
        intent_stats = {}
        total_keywords = len(df)
        
        for intent in available_intent_columns:
            count = df[intent].sum()  # Suma wartości True
            percentage = (count / total_keywords) * 100
            intent_stats[intent] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Sprawdzanie, ile słów ma przypisaną więcej niż jedną intencję
        multiple_intents_count = df.apply(
            lambda row: sum(row[col] for col in available_intent_columns if col in row),
            axis=1
        )
        
        keywords_with_multiple_intents = (multiple_intents_count > 1).sum()
        stats['mixed_intent_keywords'] = {
            'count': int(keywords_with_multiple_intents),
            'percentage': round((keywords_with_multiple_intents / total_keywords) * 100, 2)
        }
        
        stats['intent_distribution'] = intent_stats
    else:
        # Używamy standardowej kolumny intent jeśli nie ma kolumn intencji z Ahrefs
        stats['intent_distribution'] = df['intent'].value_counts().to_dict()
    
    return stats

def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Wczytuje i przygotowuje dane z pliku eksportu Ahrefs.
    
    Args:
        file_path: Ścieżka do pliku
        
    Returns:
        Tuple zawierający DataFrame z danymi i słownik ze statystykami
    """
    # Wczytaj dane
    raw_df = load_ahrefs_export(file_path)
    
    # Przetwórz dane
    processed_df = preprocess_ahrefs_data(raw_df)
    
    # Generuj statystyki
    stats = get_summary_stats(processed_df)
    
    return processed_df, stats

def save_analysis(analysis_data: Dict[str, Any], save_dir: str = "data/output") -> str:
    """
    Zapisuje kompletną analizę do pliku.
    
    Args:
        analysis_data: Słownik zawierający wszystkie dane analizy z sesji
        save_dir: Katalog do zapisu pliku
        
    Returns:
        Ścieżka do zapisanego pliku
    """
    logger.info("Zapisywanie analizy...")
    
    # Stwórz katalog jeśli nie istnieje
    os.makedirs(save_dir, exist_ok=True)
    
    # Przygotuj dane do zapisu
    save_data = {
        'metadata': {
            'save_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        },
        'data': {}
    }
    
    # Przetwarzaj każdy element z sesji
    for key, value in analysis_data.items():
        if isinstance(value, pd.DataFrame):
            # DataFrame zapisujemy jako dict i ndarrays oddzielnie
            save_data['data'][key] = {
                'type': 'DataFrame',
                'data': value.to_dict(),
                'index': value.index.tolist(),
                'columns': value.columns.tolist()
            }
        elif isinstance(value, np.ndarray):
            # NumPy arrays zapisujemy jako listy
            save_data['data'][key] = {
                'type': 'ndarray',
                'data': value.tolist(),
                'shape': value.shape,
                'dtype': str(value.dtype)
            }
        elif hasattr(value, 'to_json') or hasattr(value, 'to_dict'):
            # Wykresy Plotly i inne obiekty z serializacją
            save_data['data'][key] = {
                'type': type(value).__name__,
                'data': value.to_json() if hasattr(value, 'to_json') else value.to_dict()
            }
        elif isinstance(value, (dict, list, str, int, float, bool)):
            # Podstawowe typy Pythona
            save_data['data'][key] = {
                'type': type(value).__name__,
                'data': value
            }
        else:
            # Nieobsługiwane typy
            save_data['data'][key] = {
                'type': 'unsupported',
                'data': str(value)
            }
    
    # Zapisz do pliku
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"keyword_analysis_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Zapisz również metadane w JSON dla czytelności
    json_filepath = filepath.replace('.pkl', '_metadata.json')
    with open(json_filepath, 'w') as f:
        json.dump(save_data['metadata'], f, indent=4)
    
    logger.info(f"Analiza zapisana do: {filepath}")
    return filepath

def load_analysis(file_path: str) -> Dict[str, Any]:
    """
    Wczytuje zapisaną analizę z pliku.
    
    Args:
        file_path: Ścieżka do pliku z zapisaną analizą
        
    Returns:
        Słownik zawierający odtworzone dane analizy
    """
    logger.info(f"Wczytywanie analizy z: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Przygotuj wynik
        analysis_data = {}
        
        # Odtwórz każdy element
        for key, item in save_data['data'].items():
            if item['type'] == 'DataFrame':
                # Odtwórz DataFrame
                df = pd.DataFrame(item['data'])
                analysis_data[key] = df
                
            elif item['type'] == 'ndarray':
                # Odtwórz NumPy array
                arr = np.array(item['data'], dtype=item['dtype'])
                analysis_data[key] = arr.reshape(item['shape'])
                
            elif item['type'] in ['dict', 'list', 'str', 'int', 'float', 'bool']:
                # Podstawowe typy Pythona
                analysis_data[key] = item['data']
                
            elif item['type'] == 'Figure':
                # Wykresy Plotly
                import plotly.graph_objects as go
                fig = go.Figure(json.loads(item['data']))
                analysis_data[key] = fig
                
            else:
                # Inne typy - ostrzeżenie
                logger.warning(f"Nieobsługiwany typ podczas odczytywania: {item['type']}")
                analysis_data[key] = item['data']
        
        logger.info("Analiza wczytana pomyślnie")
        return analysis_data
        
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania analizy: {e}")
        raise