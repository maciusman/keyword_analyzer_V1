"""
Aplikacja Streamlit do analizy słów kluczowych.
"""
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Nieużywane
# import plotly.express as px # Importowane w visualizer
# import plotly.graph_objects as go # Importowane w visualizer
import os
import sys
import time
import requests # Do komunikacji z API OpenRouter
import logging  # Do logowania
from typing import List, Dict, Any, Optional, Tuple
import umap # <<<< DODANY IMPORT DLA UMAP >>>>

# Dodaj katalog projektu do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importy z projektu
from utils.data_loader import load_and_prepare_data, save_analysis, load_analysis
from models.embeddings import EmbeddingGenerator
from models.clustering import KeywordClusterer
from models.analyzer import KeywordAnalyzer
from utils.visualizer import KeywordVisualizer
from config.config import (
    DATA_DIR, OUTPUT_DIR,
    # Załóżmy, że te parametry istnieją w config.py dla wizualizacji UMAP 3D
    # Jeśli nie, możesz je zdefiniować na sztywno poniżej lub dodać do config.py
    UMAP_N_NEIGHBORS as UMAP_VIS_N_NEIGHBORS_DEFAULT, 
    UMAP_MIN_DIST as UMAP_VIS_MIN_DIST_DEFAULT,
    # Stałe OpenRouter z config.py
    OPENROUTER_API_BASE,
    OPENROUTER_MODELS_ENDPOINT,
    OPENROUTER_CHAT_ENDPOINT,
    GEMINI_MODEL as DEFAULT_GEMINI_MODEL
)

# Inicjalizacja loggera
logger = logging.getLogger(__name__)

# Konfiguracja strony
st.set_page_config(
    page_title="Analizator Słów Kluczowych",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funkcje Pomocnicze ---
@st.cache_data(ttl=3600) # Cache'uj listę modeli przez godzinę
def fetch_openrouter_models(api_key: str) -> List[str]:
    """Pobiera listę dostępnych modeli z API OpenRouter."""
    if not api_key:
        # Nie wyświetlamy błędu tutaj, bo może to być celowe (użycie .env)
        # logger.warning("Próba pobrania modeli OpenRouter bez klucza API.")
        return []
    
    headers = {"Authorization": f"Bearer {api_key}"}
    models_url = OPENROUTER_API_BASE + OPENROUTER_MODELS_ENDPOINT
    model_ids = []
    try:
        response = requests.get(models_url, headers=headers, timeout=15) # Dodano timeout
        response.raise_for_status() 
        models_data = response.json().get("data", [])
        model_ids = sorted([model.get("id") for model in models_data if model.get("id")])
        logger.info(f"Pobrano {len(model_ids)} modeli z OpenRouter.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Błąd podczas pobierania modeli z OpenRouter: {e}")
        st.error(f"Błąd połączenia z OpenRouter: {e}. Sprawdź klucz API i połączenie.")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas przetwarzania modeli OpenRouter: {e}")
        st.error(f"Wystąpił nieoczekiwany błąd: {e}")
    return model_ids

# Tytuł aplikacji
st.title("Analizator Słów Kluczowych")
st.markdown("**Narzędzie do klastrowania i analizy słów kluczowych z wykorzystaniem AI**")

# Panel boczny
st.sidebar.header("Ustawienia")

st.sidebar.subheader("Import/Eksport analizy")
if st.sidebar.button("💾 Zapisz analizę", type="secondary"):
    if 'clustered_df' in st.session_state:
        analysis_data = {
            'keywords_df': st.session_state.get('keywords_df'),
            'stats': st.session_state.get('stats'),
            'df_with_embeddings': st.session_state.get('df_with_embeddings'),
            'reduced_embeddings': st.session_state.get('reduced_embeddings'), 
            'clustered_df': st.session_state.get('clustered_df'),
            'cluster_names': st.session_state.get('cluster_names'),
            'cluster_analyses': st.session_state.get('cluster_analyses'),
            'plots': st.session_state.get('plots'),
            'intent_plot': st.session_state.get('intent_plot'),
            'cluster_map': st.session_state.get('cluster_map'),
            # <<<< DODAJEMY NOWE ELEMENTY DO ZAPISU DLA MAPY 3D >>>>
            'cluster_map_3d': st.session_state.get('cluster_map_3d'),
            'original_embeddings_for_vis': st.session_state.get('original_embeddings_for_vis'),
            'reduced_embeddings_3d_for_vis': st.session_state.get('reduced_embeddings_3d_for_vis')
        }
        save_path = save_analysis(analysis_data)
        st.sidebar.success(f"Analiza zapisana: {os.path.basename(save_path)}")
    else:
        st.sidebar.error("Brak danych do zapisania. Wykonaj analizę przed zapisaniem.")

import_file = st.sidebar.file_uploader("Wczytaj zapisaną analizę (.pkl)", type=["pkl"], key="import_analysis_file")
if import_file and 'file_already_imported' not in st.session_state:
    try:
        temp_path = os.path.join(DATA_DIR, import_file.name)
        with open(temp_path, "wb") as f: f.write(import_file.getbuffer())
        loaded_data = load_analysis(temp_path)
        for key, value in loaded_data.items(): st.session_state[key] = value
        st.session_state.analysis_completed = True
        st.session_state.file_imported = True
        st.session_state.imported_file_name = import_file.name
        st.session_state.file_already_imported = True
        st.sidebar.success(f"Analiza wczytana: {import_file.name}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Błąd podczas wczytywania analizy: {e}")
        if 'file_already_imported' in st.session_state: del st.session_state.file_already_imported

st.sidebar.markdown("---")

# <<<< POCZĄTEK NOWEJ SEKCJI: Wybór Modelu AI i Klucze API >>>>
st.sidebar.markdown("---") # Dodatkowe oddzielenie
st.sidebar.subheader("Konfiguracja Modelu AI")

# Inicjalizacja stanu sesji
if 'ai_provider' not in st.session_state: st.session_state.ai_provider = "Gemini"
if 'selected_openrouter_model' not in st.session_state: st.session_state.selected_openrouter_model = None
if 'openrouter_models_list' not in st.session_state: st.session_state.openrouter_models_list = []
# Przechowuj klucze w session_state, aby nie znikały po interakcjach
if 'gemini_key_ui' not in st.session_state: st.session_state.gemini_key_ui = ""
if 'openrouter_key_ui' not in st.session_state: st.session_state.openrouter_key_ui = ""
if 'jina_key_ui' not in st.session_state: st.session_state.jina_key_ui = ""

# Wybór dostawcy
ai_provider = st.sidebar.radio(
    "Wybierz dostawcę modelu językowego:",
    ("Gemini", "OpenRouter"),
    key='ai_provider_radio',
    index=["Gemini", "OpenRouter"].index(st.session_state.ai_provider), # Ustaw domyślny
    # Nie potrzeba on_change, bo stan jest aktualizowany przez sam widget
)
st.session_state.ai_provider = ai_provider # Zapisz wybór w stanie sesji

st.sidebar.subheader("Klucze API")
st.sidebar.caption("Wprowadź klucze API. Jeśli pozostawisz puste, aplikacja spróbuje użyć kluczy z pliku .env (jeśli działasz lokalnie).")

# Pola na klucze API - przypisanie do session_state
st.session_state.gemini_key_ui = st.sidebar.text_input(
    "Klucz API Google Gemini:", 
    type="password", 
    value=st.session_state.gemini_key_ui, # Odczytaj wartość ze stanu
    help="Potrzebny, jeśli jako dostawcę wybrano Gemini."
)
st.session_state.openrouter_key_ui = st.sidebar.text_input(
    "Klucz API OpenRouter:", 
    type="password", 
    value=st.session_state.openrouter_key_ui, # Odczytaj wartość ze stanu
    help="Potrzebny, jeśli jako dostawcę wybrano OpenRouter."
)
st.session_state.jina_key_ui = st.sidebar.text_input(
    "Klucz API JINA:", 
    type="password", 
    value=st.session_state.jina_key_ui, # Odczytaj wartość ze stanu
    help="Potrzebny do generowania embeddingów."
)

# Sekcja dla OpenRouter - ładowanie i wybór modelu
if st.session_state.ai_provider == "OpenRouter":
    st.sidebar.markdown("### Konfiguracja OpenRouter")
    
    # Przycisk do pobrania modeli
    if st.sidebar.button("Pobierz/Odśwież modele OpenRouter"):
        or_key = st.session_state.openrouter_key_ui.strip()
        if or_key:
            # Wywołaj funkcję z cache - jeśli klucz się nie zmienił, zwróci z cache
            with st.spinner("Pobieranie listy modeli..."):
                models = fetch_openrouter_models(or_key) 
            if models:
                st.session_state.openrouter_models_list = models
                # Jeśli poprzednio wybrany model nie jest już na liście lub nic nie wybrano,
                # ustaw domyślny (np. pierwszy z listy) lub zostaw None
                if st.session_state.selected_openrouter_model not in models:
                    st.session_state.selected_openrouter_model = models[0] if models else None
                st.sidebar.success(f"Pobrano/odświeżono {len(models)} modeli.")
                st.rerun() # Przeładuj, aby zaktualizować selectbox
            else:
                st.session_state.openrouter_models_list = [] 
                st.session_state.selected_openrouter_model = None # Resetuj wybór
        else:
            st.sidebar.warning("Wprowadź klucz API OpenRouter, aby pobrać listę modeli.")

    # Selectbox do wyboru modelu OpenRouter
    # Upewnij się, że indeks jest poprawny
    current_model_index = 0
    if st.session_state.selected_openrouter_model and st.session_state.openrouter_models_list:
        try:
            current_model_index = st.session_state.openrouter_models_list.index(st.session_state.selected_openrouter_model)
        except ValueError:
            current_model_index = 0 # Jeśli poprzedni wybór zniknął, wybierz pierwszy

    selected_model = st.sidebar.selectbox(
        "Wybierz model OpenRouter:",
        st.session_state.openrouter_models_list,
        index=current_model_index,
        key='selected_model_selectbox_widget', # Użyj innego klucza dla widgetu
        disabled=not st.session_state.openrouter_models_list 
    )
    # Zapisz wybór do stanu sesji tylko jeśli jest różny
    if selected_model != st.session_state.selected_openrouter_model:
         st.session_state.selected_openrouter_model = selected_model
         # Nie ma potrzeby rerun tutaj, selectbox sam się aktualizuje
# <<<< KONIEC NOWEJ SEKCJI >>>>

st.sidebar.markdown("---")
st.sidebar.subheader("Nowa analiza")

def process_keywords(file_path):
    progress_col, status_col = st.columns([3, 2])
    overall_progress_bar = progress_col.progress(0)
    overall_status = progress_col.empty()
    step_progress_bar = progress_col.progress(0)
    step_status = progress_col.empty()
    current_step_container = status_col.empty()
    
    def update_progress(overall_progress, step_progress, overall_text, step_text, step_name):
        overall_progress_bar.progress(overall_progress)
        overall_status.text(overall_text)
        step_progress_bar.progress(step_progress)
        step_status.text(step_text)
        current_step_container.markdown(f"### Obecny krok: {step_name}")
    
    update_progress(0.0, 0.0, "Inicjalizacja...", "Wczytywanie pliku...", "Przetwarzanie danych")
    df, stats = load_and_prepare_data(file_path)
    st.session_state.keywords_df = df
    st.session_state.stats = stats
    update_progress(0.05, 1.0, "Przetwarzanie danych zakończone", "Plik wczytany", "Przetwarzanie danych")
    
    # Krok 2: Generuj embeddingi (10% - 30%)
    # <<<< ZMIANA: Inicjalizacja z kluczem JINA z session_state >>>>
    current_jina_api_key = st.session_state.jina_key_ui.strip() if st.session_state.jina_key_ui and st.session_state.jina_key_ui.strip() else None
    try:
        embedding_generator = EmbeddingGenerator(api_key=current_jina_api_key)
        # Sprawdzanie czy klient powstał (zakładając, że konstruktor ustawia self.client na None przy błędzie)
        if hasattr(embedding_generator, 'client') and embedding_generator.client is None:
            # Spróbujmy załadować z config jako fallback, jeśli nie podano w UI
            from config.config import JINA_API_KEY as CONFIG_JINA_API_KEY
            if CONFIG_JINA_API_KEY:
                embedding_generator = EmbeddingGenerator(api_key=CONFIG_JINA_API_KEY)
                if hasattr(embedding_generator, 'client') and embedding_generator.client is None:
                    st.error("Klucz API JINA z pliku .env również jest nieprawidłowy.")
                    return
            else:
                 st.error("Brak klucza API dla JINA. Podaj go w panelu bocznym lub ustaw w .env.")
                 return
    except Exception as e: 
        st.error(f"Błąd inicjalizacji JINA: {e}")
        return
    # <<<< KONIEC ZMIANY >>>>
    
    def embedding_callback(progress, status):
        update_progress(0.05 + 0.20 * progress, progress, "Generowanie embedingów...", status, "Generowanie embedingów")
    df_with_embeddings = embedding_generator.process_keywords_dataframe(df, progress_callback=embedding_callback)
    st.session_state.df_with_embeddings = df_with_embeddings
    
    # <<<< PRZECHOWUJEMY ORYGINALNE EMBEDDINGI DLA WIZUALIZACJI 3D >>>>
    if 'embedding' in df_with_embeddings.columns:
        st.session_state.original_embeddings_for_vis = np.array(df_with_embeddings['embedding'].tolist())
    else:
        st.warning("Brak kolumny 'embedding' w danych. Mapa 3D może nie zostać wygenerowana.")
        st.session_state.original_embeddings_for_vis = None

    clusterer = KeywordClusterer()
    def clustering_callback(progress, status):
        update_progress(0.25 + 0.20 * progress, progress, "Klastrowanie słów kluczowych...", status, "Klastrowanie")
    clustered_df = clusterer.process_keywords_dataframe(df_with_embeddings, progress_callback=clustering_callback)
    st.session_state.reduced_embeddings = clusterer.reduced_embeddings 
    st.session_state.clustered_df = clustered_df
    
    # Krok 4: Nazwij klastry (50% - 65%)
    # <<<< ZMIANA: Inicjalizacja Analizatora z odpowiednim kluczem i modelem >>>>
    llm_api_key_to_use = None
    llm_model_name_to_use = None
    llm_provider = st.session_state.ai_provider # Pobierz wybranego dostawcę

    if llm_provider == "Gemini":
        # Sprawdź klucz z UI, potem z config/.env
        key_from_ui = st.session_state.gemini_key_ui.strip()
        if key_from_ui:
            llm_api_key_to_use = key_from_ui
        else:
            from config.config import GEMINI_API_KEY as CONFIG_GEMINI_API_KEY
            if CONFIG_GEMINI_API_KEY:
                llm_api_key_to_use = CONFIG_GEMINI_API_KEY
            else:
                 st.error("Wybrano Gemini, ale nie podano klucza API ani nie znaleziono go w .env.")
                 return # Zakończ, jeśli nie ma klucza

        llm_model_name_to_use = DEFAULT_GEMINI_MODEL # Użyj domyślnego Gemini z config
    
    elif llm_provider == "OpenRouter":
        # Sprawdź klucz z UI, potem z config/.env
        key_from_ui = st.session_state.openrouter_key_ui.strip()
        if key_from_ui:
            llm_api_key_to_use = key_from_ui
        else:
            from config.config import OPENROUTER_API_KEY as CONFIG_OPENROUTER_API_KEY
            if CONFIG_OPENROUTER_API_KEY:
                llm_api_key_to_use = CONFIG_OPENROUTER_API_KEY
            else:
                 st.error("Wybrano OpenRouter, ale nie podano klucza API ani nie znaleziono go w .env.")
                 return # Zakończ, jeśli nie ma klucza
        
        # Pobierz wybrany model ze stanu sesji
        if 'selected_openrouter_model' not in st.session_state:
            st.session_state.selected_openrouter_model = None  # Inicjalizacja, jeśli nie istnieje
        llm_model_name_to_use = st.session_state.selected_openrouter_model 
        if not llm_model_name_to_use:
            st.error("Wybrano OpenRouter, ale nie wybrano modelu. Pobierz listę modeli i wybierz jeden.")
            return
    
    # Inicjalizuj KeywordAnalyzer z wybranymi parametrami
    try:
        analyzer = KeywordAnalyzer(
            api_key=llm_api_key_to_use, 
            model_name=llm_model_name_to_use,
            provider=llm_provider # Przekaż informację o dostawcy
        )
    except ValueError as e: 
        st.error(f"Błąd inicjalizacji modelu AI: {e}")
        return
    except Exception as e:
        st.error(f"Nieoczekiwany błąd inicjalizacji modelu AI: {e}")
        return
    # <<<< KONIEC ZMIANY >>>>
    
    def naming_callback(progress, status):
        update_progress(0.45 + 0.15 * progress, progress, "Nazywanie klastrów...", status, "Nazywanie klastrów")
    cluster_names = analyzer.name_clusters(clustered_df, progress_callback=naming_callback)
    st.session_state.cluster_names = cluster_names
    
    def analysis_callback(progress, status):
        update_progress(0.60 + 0.18 * progress, progress, "Analizowanie klastrów...", status, "Analiza klastrów") # Zmieniono overall_progress
    cluster_analyses = analyzer.process_all_clusters(clustered_df, cluster_names, progress_callback=analysis_callback)
    st.session_state.cluster_analyses = cluster_analyses
    
    # <<<< GENEROWANIE EMBEDDINGÓW 3D DLA WIZUALIZACJI >>>>
    update_progress(0.78, 0.0, "Przygotowanie danych do wizualizacji 3D...", "Redukcja do 3D...", "Wizualizacja 3D")
    if st.session_state.get('original_embeddings_for_vis') is not None:
        try:
            umap_3d_visualizer = umap.UMAP(
                n_neighbors=UMAP_VIS_N_NEIGHBORS_DEFAULT, 
                n_components=3, 
                min_dist=UMAP_VIS_MIN_DIST_DEFAULT,    
                random_state=42,
                n_jobs=1 
            )
            reduced_embeddings_3d_for_vis = umap_3d_visualizer.fit_transform(st.session_state.original_embeddings_for_vis)
            st.session_state.reduced_embeddings_3d_for_vis = reduced_embeddings_3d_for_vis
            update_progress(0.80, 1.0, "Dane do wizualizacji 3D gotowe.", "Redukcja do 3D zakończona.", "Wizualizacja 3D")
        except Exception as e:
            st.warning(f"Nie udało się wygenerować embeddingów 3D dla wizualizacji: {e}")
            st.session_state.reduced_embeddings_3d_for_vis = None
            update_progress(0.80, 1.0, "Błąd w przygotowaniu danych 3D.", f"Błąd: {e}", "Wizualizacja 3D")
    else:
        update_progress(0.80, 1.0, "Pominięto przygotowanie danych 3D.", "Brak embeddingów.", "Wizualizacja 3D")
        st.session_state.reduced_embeddings_3d_for_vis = None


    update_progress(0.85, 0.0, "Tworzenie wizualizacji...", "Przygotowywanie wykresów...", "Wizualizacja")
    visualizer = KeywordVisualizer()
    
    update_progress(0.88, 0.2, "Tworzenie wizualizacji...", "Wykres podsumowania klastrów...", "Wizualizacja")
    plots = visualizer.plot_cluster_summary(cluster_analyses)
    st.session_state.plots = plots
    
    update_progress(0.91, 0.4, "Tworzenie wizualizacji...", "Wykres rozkładu intencji...", "Wizualizacja")
    intent_plot = visualizer.plot_intent_distribution(stats)
    st.session_state.intent_plot = intent_plot
    
    update_progress(0.94, 0.6, "Tworzenie wizualizacji...", "Mapa klastrów 2D...", "Wizualizacja")
    cluster_map_2d = visualizer.plot_cluster_map(clustered_df, cluster_names, st.session_state.reduced_embeddings)
    st.session_state.cluster_map = cluster_map_2d
    
    # <<<< GENEROWANIE MAPY 3D >>>>
    update_progress(0.97, 0.8, "Tworzenie wizualizacji...", "Mapa klastrów 3D...", "Wizualizacja")
    if st.session_state.get('reduced_embeddings_3d_for_vis') is not None:
        cluster_map_3d_fig = visualizer.plot_cluster_map_3d(
            st.session_state.clustered_df, 
            st.session_state.cluster_names, 
            st.session_state.reduced_embeddings_3d_for_vis
        )
        st.session_state.cluster_map_3d = cluster_map_3d_fig
    else:
        st.session_state.cluster_map_3d = None
        # st.warning("Pominięto generowanie mapy klastrów 3D z powodu braku embeddingów 3D.") # Ostrzeżenie już było wyżej

    update_progress(1.0, 1.0, "Analiza zakończona", "Wszystkie wizualizacje gotowe", "Zakończono")
    time.sleep(0.5)
    st.success("Analiza zakończona!")
    progress_col.empty(); status_col.empty()

upload_file = st.sidebar.file_uploader("Wczytaj plik CSV/Excel z Ahrefs", type=["csv", "xlsx", "xls"])
if upload_file:
    file_path = os.path.join(DATA_DIR, upload_file.name)
    with open(file_path, "wb") as f: f.write(upload_file.getbuffer())
    st.session_state.file_uploaded = True
    st.session_state.file_path = file_path
    st.session_state.file_name = upload_file.name
    st.sidebar.success(f"Plik wczytany: {upload_file.name}")
    
    if st.sidebar.button("🚀 Uruchom analizę", type="primary"):
        keys_to_clear = ['clustered_df', 'cluster_analyses', 'plots', 'intent_plot', 'cluster_map', 
                 'keywords_df', 'stats', 'df_with_embeddings', 'reduced_embeddings', 
                 'cluster_names', 
                 'cluster_map_3d', 'original_embeddings_for_vis', 'reduced_embeddings_3d_for_vis',
                 'analysis_completed', 'file_imported'
                 # NIE czyścimy zmiennych OpenRouter
                ]
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        if 'file_already_imported' in st.session_state: del st.session_state.file_already_imported # Resetujemy flagę importu pliku
        process_keywords(file_path)
        st.session_state.analysis_completed = True
        
    if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
        if st.sidebar.button("↻ Analizuj ponownie"):
            keys_to_clear_rerun = ['clustered_df', 'cluster_analyses', 'plots', 'intent_plot', 'cluster_map', 
                       'keywords_df', 'stats', 'df_with_embeddings', 'reduced_embeddings', 
                       'cluster_names', 
                       'cluster_map_3d', 'original_embeddings_for_vis', 'reduced_embeddings_3d_for_vis',
                       'analysis_completed', 'file_imported'
                       # NIE czyścimy zmiennych OpenRouter
                      ]
            for key in keys_to_clear_rerun:
                if key not in ['file_uploaded', 'file_path', 'file_name'] and key in st.session_state:
                    del st.session_state[key]
            if 'file_already_imported' in st.session_state: del st.session_state.file_already_imported # Resetujemy flagę importu pliku
            process_keywords(file_path)
            st.session_state.analysis_completed = True


if 'clustered_df' in st.session_state:
    st.header("Przegląd danych")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Liczba słów kluczowych", st.session_state.stats.get('total_keywords',0))
    with col2: st.metric("Łączny wolumen wyszukiwań", st.session_state.stats.get('total_volume',0))
    with col3: st.metric("Średnia trudność (KD)", f"{st.session_state.stats.get('avg_difficulty',0.0):.2f}")
    
    st.header("Rozkład intencji wyszukiwania")
    if 'intent_plot' in st.session_state and st.session_state.intent_plot:
        st.plotly_chart(st.session_state.intent_plot, use_container_width=True)
    
    st.header("Mapa klastrów słów kluczowych (2D)")
    if 'cluster_map' in st.session_state and st.session_state.cluster_map:
        st.plotly_chart(st.session_state.cluster_map, use_container_width=True)

    # <<<< NOWA SEKCJA: WYŚWIETLANIE MAPY KLASTRÓW 3D >>>>
    if 'cluster_map_3d' in st.session_state and st.session_state.cluster_map_3d is not None:
        st.header("Interaktywna Mapa Klastrów Słów Kluczowych (3D)")
        # Używamy st.container() z ustaloną wysokością dla wykresu 3D, aby uniknąć nadmiernego scrollowania
        with st.container():
            st.plotly_chart(st.session_state.cluster_map_3d, use_container_width=True, height=700) 
    elif st.session_state.get('analysis_completed'): 
        if not st.session_state.get('original_embeddings_for_vis'):
             st.warning("Nie można wygenerować mapy 3D: Brak oryginalnych embeddingów.")
        elif not st.session_state.get('reduced_embeddings_3d_for_vis') and st.session_state.get('original_embeddings_for_vis') is not None:
             st.warning("Nie udało się wygenerować mapy klastrów 3D (prawdopodobnie błąd podczas redukcji wymiarowości).")
    # <<<< KONIEC NOWEJ SEKCJI >>>>

    st.header("Podsumowanie klastrów")
    if 'plots' in st.session_state and st.session_state.plots:
        col1_sum, col2_sum = st.columns(2)
        with col1_sum:
            st.subheader("Wolumen wyszukiwań według klastrów")
            st.plotly_chart(st.session_state.plots['volume'], use_container_width=True)
        with col2_sum:
            st.subheader("Liczba słów kluczowych według klastrów")
            st.plotly_chart(st.session_state.plots['count'], use_container_width=True)
        st.subheader("Trudność vs. wolumen vs. liczba słów kluczowych")
        st.plotly_chart(st.session_state.plots['scatter'], use_container_width=True)
        st.subheader("Rozkład priorytetów klastrów")
        st.plotly_chart(st.session_state.plots['priority'], use_container_width=True)
    
    st.header("Szczegółowa analiza klastrów")
    if 'cluster_analyses' in st.session_state:
        for cluster_analysis in st.session_state.cluster_analyses:
            priority_color = "#66bb6a"
            if cluster_analysis.get('priority_level') == 'Wysoki': priority_color = "#ff7043"
            elif cluster_analysis.get('priority_level') == 'Średni': priority_color = "#ffa726"
            
            st.markdown(f"""<h3 style="margin-bottom: 0px;">{cluster_analysis.get('cluster_name', 'Brak Nazwy')} <span style="color: {priority_color}; font-size: 0.8em;">[{cluster_analysis.get('priority_level', 'Nieznany')} priorytet]</span></h3>""", unsafe_allow_html=True)
            
            # Poprawiona kolejność wyświetlania (zgodnie z Twoim screenem)
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5) 
            with col_m1: st.metric("Liczba słów kluczowych", cluster_analysis.get('keywords_count', 0))
            with col_m2: st.metric("Łączny wolumen", cluster_analysis.get('total_volume', 0))
            with col_m3: st.metric("Średnia trudność", f"{cluster_analysis.get('avg_difficulty', 0.0):.2f}")
            with col_m4: st.metric("Średni CPC", f"${cluster_analysis.get('avg_cpc', 0.0):.2f}")
            with col_m5: st.metric("Wynik Priorytetu", f"{cluster_analysis.get('priority_score', 0.0):.2f}")

            st.markdown(f"**Uzasadnienie Priorytetu (ocena LLM):** {cluster_analysis.get('priority_justification', 'Brak uzasadnienia od modelu.')}")
            st.markdown("**Wnioski (Insights):**")
            st.write(cluster_analysis.get('insights', 'Brak wniosków.'))
            st.markdown("**Strategia Contentowa (Content Strategy):**")
            st.write(cluster_analysis.get('content_strategy', 'Brak strategii.'))
            
            with st.expander("Zobacz słowa kluczowe w tym klastrze"):
                cluster_id = cluster_analysis.get('cluster_id')
                if cluster_id is not None and 'clustered_df' in st.session_state:
                    cluster_keywords_df_exp = st.session_state.clustered_df[st.session_state.clustered_df['cluster'] == cluster_id]
                    if not cluster_keywords_df_exp.empty:
                        cluster_keywords_df_exp = cluster_keywords_df_exp.sort_values(by='volume', ascending=False) if 'volume' in cluster_keywords_df_exp else cluster_keywords_df_exp
                        intent_cols_exp = ['branded', 'local', 'navigational', 'informational', 'commercial', 'transactional']
                        available_intent_cols_exp = [col for col in intent_cols_exp if col in cluster_keywords_df_exp.columns]
                        display_cols_exp = ['keyword'] + [col for col in ['volume', 'difficulty', 'cpc'] if col in cluster_keywords_df_exp.columns]
                        if available_intent_cols_exp: display_cols_exp.extend(available_intent_cols_exp)
                        elif 'intent' in cluster_keywords_df_exp.columns: display_cols_exp.append('intent')
                        st.dataframe(cluster_keywords_df_exp[display_cols_exp])
                    else:
                        st.write("Brak słów kluczowych dla tego ID klastra.")
                else:
                    st.write("Nie można załadować słów kluczowych dla tego klastra.")
            st.markdown("---")
    
    st.header("Eksport wyników")
    if st.button("Eksportuj wszystkie słowa kluczowe z klastrami"):
        export_df_keywords = st.session_state.clustered_df.copy()
        export_df_keywords['cluster_name'] = export_df_keywords['cluster'].map(lambda c: st.session_state.cluster_names.get(c, f"Cluster {c}") if c != -1 else "Outliers")
        cols_to_drop_exp = [col for col in ['embedding'] if col in export_df_keywords.columns] 
        if 'intent_list' in export_df_keywords.columns and not export_df_keywords.empty and hasattr(export_df_keywords['intent_list'].iloc[0], '__iter__') and not isinstance(export_df_keywords['intent_list'].iloc[0], str): # Lepsze sprawdzenie listy
             export_df_keywords['intent_list'] = export_df_keywords['intent_list'].apply(lambda x: ', '.join(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x)
        if cols_to_drop_exp: export_df_keywords = export_df_keywords.drop(columns=cols_to_drop_exp)
        export_path_keywords = os.path.join(OUTPUT_DIR, "keywords_with_clusters.csv")
        export_df_keywords.to_csv(export_path_keywords, index=False)
        st.markdown(f"Plik został zapisany w: `{export_path_keywords}`")
    
    if st.button("Eksportuj analizę klastrów"):
        export_df_analysis = pd.DataFrame(st.session_state.cluster_analyses)
        export_path_analysis = os.path.join(OUTPUT_DIR, "cluster_analysis.csv")
        export_df_analysis.to_csv(export_path_analysis, index=False)
        st.markdown(f"Plik został zapisany w: `{export_path_analysis}`")
else:
    if 'file_imported' in st.session_state and st.session_state.file_imported:
        st.info(f"Wczytano zapisaną analizę: {st.session_state.imported_file_name}")
    else:
        st.info("Wczytaj plik CSV lub Excel z eksportu Ahrefs, aby rozpocząć analizę, lub zaimportuj zapisaną analizę.")
    st.markdown("""
    ### Oczekiwana struktura danych
    Plik powinien zawierać co najmniej następujące kolumny: `keyword`, `volume`, `difficulty` / `kd`, `intent`, `cpc`.
    Dodatkowo, dla rozkładu intencji, aplikacja obsługuje kolumny: `branded`, `local`, `navigational`, `informational`, `commercial`, `transactional`.
    Nazwy kolumn mogą nieznacznie się różnić - aplikacja spróbuje je zmapować automatycznie.
    """)