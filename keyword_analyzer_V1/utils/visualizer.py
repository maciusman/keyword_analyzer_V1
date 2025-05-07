"""
Moduł odpowiedzialny za wizualizację wyników analizy słów kluczowych.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Nie jest używane, można usunąć jeśli nie planujesz
import seaborn as sns # Nie jest używane, można usunąć jeśli nie planujesz
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import os

# Konfiguracja logowania
# Zakładamy, że jest już skonfigurowane w main.py lub app.py
# Jeśli nie, odkomentuj:
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordVisualizer:
    """
    Klasa odpowiedzialna za wizualizację wyników analizy słów kluczowych.
    """
    
    def __init__(self, output_dir: str = "data/output"):
        """
        Inicjalizuje wizualizator.
        
        Args:
            output_dir: Katalog docelowy dla plików wizualizacji
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Poprawka: użyj self.output_dir
        logger.info(f"Zainicjalizowano KeywordVisualizer z katalogiem wyjściowym: {self.output_dir}")
    
    def plot_cluster_summary(self, cluster_analyses: List[Dict[str, Any]], top_n: int = 15) -> Dict[str, Any]:
        """
        Tworzy podsumowanie klastrów w postaci wykresów.
        
        Args:
            cluster_analyses: Lista słowników zawierających wyniki analizy klastrów
            top_n: Liczba najważniejszych klastrów do uwzględnienia w wykresach
            
        Returns:
            Słownik zawierający obiekty wykresów Plotly
        """
        logger.info(f"Tworzenie podsumowania klastrów (top {top_n})...")
        
        if not cluster_analyses:
            logger.warning("Brak danych analizy klastrów do wygenerowania podsumowania.")
            # Zwróć puste figury, aby uniknąć błędów w UI
            empty_fig = go.Figure()
            empty_fig.update_layout(title_text="Brak danych do podsumowania klastrów")
            return {
                'volume': empty_fig,
                'count': empty_fig,
                'scatter': empty_fig,
                'priority': empty_fig
            }

        df = pd.DataFrame(cluster_analyses)
        
        if df.empty or 'priority_score' not in df.columns:
            logger.warning("DataFrame analizy klastrów jest pusty lub brakuje kolumny 'priority_score'.")
            empty_fig = go.Figure()
            empty_fig.update_layout(title_text="Niekompletne dane do podsumowania klastrów")
            return {
                'volume': empty_fig,
                'count': empty_fig,
                'scatter': empty_fig,
                'priority': empty_fig
            }

        top_clusters = df.sort_values(by='priority_score', ascending=False).head(top_n)
        
        fig_volume = px.bar(
            top_clusters, x='cluster_name', y='total_volume',
            title='Wolumen wyszukiwań według klastrów słów kluczowych',
            labels={'cluster_name': 'Klaster', 'total_volume': 'Całkowity wolumen wyszukiwań'},
            color='priority_level',
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a', 'Nieznany': '#9e9e9e'},
            template='plotly_white'
        )
        fig_volume.update_layout(xaxis_tickangle=-45)
        
        fig_count = px.bar(
            top_clusters, x='cluster_name', y='keywords_count',
            title='Liczba słów kluczowych według klastrów',
            labels={'cluster_name': 'Klaster', 'keywords_count': 'Liczba słów kluczowych'},
            color='priority_level',
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a', 'Nieznany': '#9e9e9e'},
            template='plotly_white'
        )
        fig_count.update_layout(xaxis_tickangle=-45)
        
        fig_scatter = px.scatter(
            df, x='avg_difficulty', y='total_volume', size='keywords_count',
            color='priority_level', hover_name='cluster_name',
            title='Trudność vs. wolumen vs. liczba słów kluczowych',
            labels={'avg_difficulty': 'Średnia trudność (KD)', 'total_volume': 'Całkowity wolumen wyszukiwań', 'keywords_count': 'Liczba słów kluczowych'},
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a', 'Nieznany': '#9e9e9e'},
            template='plotly_white'
        )
        
        priority_counts = df['priority_level'].value_counts().reset_index()
        priority_counts.columns = ['priority_level', 'count']
        fig_priority = px.pie(
            priority_counts, values='count', names='priority_level',
            title='Rozkład priorytetów klastrów',
            color='priority_level',
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a', 'Nieznany': '#9e9e9e'},
            template='plotly_white'
        )
        
        try:
            output_prefix = os.path.join(self.output_dir, "cluster")
            fig_volume.write_html(f"{output_prefix}_volume.html")
            fig_count.write_html(f"{output_prefix}_count.html")
            fig_scatter.write_html(f"{output_prefix}_scatter.html")
            fig_priority.write_html(f"{output_prefix}_priority.html")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania wykresów podsumowania klastrów: {e}")
            
        return {'volume': fig_volume, 'count': fig_count, 'scatter': fig_scatter, 'priority': fig_priority}
    
    def plot_intent_distribution(self, stats: Dict[str, Any]) -> go.Figure:
        logger.info("Tworzenie wykresu rozkładu intencji...")
        intent_distribution = stats.get('intent_distribution', {})
        if not intent_distribution:
            logger.warning("Brak danych o intencjach. Tworzę pusty wykres.")
            fig = go.Figure().update_layout(title_text='Brak danych o intencjach wyszukiwania', template='plotly_white')
            return fig
        
        intents, counts, percentages = [], [], []
        for intent, data in intent_distribution.items():
            if isinstance(data, dict) and 'count' in data and 'percentage' in data:
                intents.append(intent); counts.append(data['count']); percentages.append(data['percentage'])
            elif isinstance(data, (int, float)) and sum(intent_distribution.values()) > 0: # Dodano sprawdzenie sumy dla starego formatu
                intents.append(intent); counts.append(int(data)); percentages.append(round((data / sum(intent_distribution.values())) * 100, 2))
            elif isinstance(data, (int, float)) and sum(intent_distribution.values()) == 0: # Obsługa gdy suma to 0
                intents.append(intent); counts.append(int(data)); percentages.append(0.0)


        color_map = {'informational': '#4caf50', 'commercial': '#ff9800', 'transactional': '#e91e63', 'navigational': '#2196f3', 'branded': '#673ab7', 'local': '#00bcd4', 'unknown': '#9e9e9e'}
        label_map = {'informational': 'Informacyjna', 'commercial': 'Komercyjna', 'transactional': 'Transakcyjna', 'navigational': 'Nawigacyjna', 'branded': 'Brandowa', 'local': 'Lokalna', 'unknown': 'Nieznana'}
        
        intent_df = pd.DataFrame({'intent': intents, 'count': counts, 'percentage': percentages, 'intent_pl': [label_map.get(str(intent).lower(), str(intent)) for intent in intents]}) # Dodano str() dla bezpieczeństwa
        
        if intent_df.empty or 'percentage' not in intent_df.columns or intent_df['percentage'].sum() == 0:
            logger.warning("DataFrame intencji jest pusty lub nie zawiera danych do wyświetlenia na wykresie kołowym.")
            fig = go.Figure().update_layout(title_text='Brak danych procentowych dla intencji', template='plotly_white')
            return fig

        fig = px.pie(
            intent_df, values='percentage', names='intent_pl', title='Rozkład intencji wyszukiwania',
            color='intent', color_discrete_map=color_map, template='plotly_white'
        )
        if 'mixed_intent_keywords' in stats and isinstance(stats['mixed_intent_keywords'], dict) and 'percentage' in stats['mixed_intent_keywords']:
            mixed_info = stats['mixed_intent_keywords']
            fig.add_annotation(text=f"Uwaga: {mixed_info['percentage']}% słów kluczowych ma więcej niż jedną intencję", xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False, font=dict(size=12))
        
        try:
            output_path = os.path.join(self.output_dir, "intent_distribution.html")
            fig.write_html(output_path)
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania wykresu rozkładu intencji: {e}")
        return fig
    
    def plot_cluster_map(self, df: pd.DataFrame, cluster_names: Dict[int, str], reduced_embeddings: Optional[np.ndarray]) -> go.Figure: # Zmieniono na Optional
        if reduced_embeddings is None or reduced_embeddings.shape[1] < 2: # Sprawdzenie None
            logger.error("Potrzebne są co najmniej 2 wymiary dla wizualizacji klastrów 2D. Embeddingi nie są dostępne lub mają zły kształt.")
            fig = go.Figure()
            fig.update_layout(title_text="Błąd: Brak danych 2D do wizualizacji mapy klastrów.")
            return fig # Zwróć pusty wykres zamiast None
        
        logger.info("Tworzenie mapy klastrów 2D...")
        viz_df = df[['keyword', 'cluster']].copy()
        viz_df['x'] = reduced_embeddings[:, 0]
        viz_df['y'] = reduced_embeddings[:, 1]
        viz_df['cluster_name'] = viz_df['cluster'].map(lambda c: cluster_names.get(c, f"Klaster {c}") if c != -1 else "Outliers")
        
        try:
            fig = px.scatter(
                viz_df, x='x', y='y', color='cluster_name', hover_name='keyword',
                title='Mapa klastrów słów kluczowych (2D)', labels={'x': 'Wymiar UMAP 1', 'y': 'Wymiar UMAP 2'},
                template='plotly_dark' # Zmieniono na ciemny szablon dla spójności z potencjalną mapą 3D
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(legend_title_text='Klaster', legend=dict(yanchor="top", y=1, xanchor="left", x=1.02), uirevision='true', dragmode=False)
            
            output_path = os.path.join(self.output_dir, "cluster_map_2d.html") # Zmieniono nazwę pliku
            fig.write_html(output_path)
            logger.info(f"Mapa klastrów 2D zapisana do: {output_path}")
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia mapy klastrów 2D: {e}")
            fig = go.Figure()
            fig.update_layout(title_text="Błąd podczas generowania mapy klastrów 2D.")
            return fig
        return fig

    # <<<< NOWA METODA DLA MAPY 3D >>>>
    def plot_cluster_map_3d(self, 
                            df: pd.DataFrame, 
                            cluster_names: Dict[int, str], 
                            reduced_embeddings_3d: Optional[np.ndarray]) -> go.Figure:
        """
        Tworzy interaktywną mapę klastrów w przestrzeni 3D.
        """
        if reduced_embeddings_3d is None or not isinstance(reduced_embeddings_3d, np.ndarray) or reduced_embeddings_3d.ndim != 2 or reduced_embeddings_3d.shape[1] != 3:
            logger.error("Nieprawidłowe dane dla wizualizacji 3D klastrów. Embeddingi 3D nie są dostępne lub mają zły kształt.")
            fig = go.Figure()
            fig.update_layout(title_text="Błąd: Brak danych 3D do wizualizacji mapy klastrów.")
            return fig
        
        logger.info("Tworzenie mapy klastrów 3D...")
        
        viz_df = df[['keyword', 'cluster']].copy()
        viz_df['x'] = reduced_embeddings_3d[:, 0]
        viz_df['y'] = reduced_embeddings_3d[:, 1]
        viz_df['z'] = reduced_embeddings_3d[:, 2]
        
        size_param = None
        hover_data_custom = {'x': False, 'y': False, 'z': False} # Ukryj współrzędne z hover

        if 'volume' in df.columns and not df['volume'].empty:
            # <<<< ZMIANA TUTAJ: Transformacja logarytmiczna wolumenu >>>>
            # Dodajemy małą stałą (np. 1), aby uniknąć log(0) lub log(wartości ujemnych)
            # i aby punkty z wolumenem 0 lub 1 nadal miały jakiś minimalny rozmiar.
            viz_df['volume_log_scaled'] = np.log1p(df['volume'].fillna(0)) # log1p(x) to log(1+x)
            
            # Opcjonalnie: normalizacja logarytmicznie przeskalowanego wolumenu do zakresu np. 1-20 dla lepszej kontroli
            # min_log_vol = viz_df['volume_log_scaled'].min()
            # max_log_vol = viz_df['volume_log_scaled'].max()
            # if max_log_vol > min_log_vol:
            #     viz_df['volume_display_size'] = 1 + 19 * (viz_df['volume_log_scaled'] - min_log_vol) / (max_log_vol - min_log_vol)
            # else:
            #     viz_df['volume_display_size'] = 5 # Domyślny rozmiar, jeśli wszystkie wartości są takie same
            # size_param = 'volume_display_size'
            
            size_param = 'volume_log_scaled' # Użyjemy bezpośrednio logarytmicznie przeskalowanego
            hover_data_custom['volume_log_scaled'] = False # Ukryj logarytmiczny wolumen z hover
            # POPRAWKA: Nie dodajemy już tutaj oryginalnego wolumenu do hover_data_custom
            
            # POPRAWKA: Dodajemy kolumnę oryginalny_wolumen do viz_df
            viz_df['oryginalny_wolumen'] = df['volume'].fillna(0).astype(int)

        viz_df['cluster_name'] = viz_df['cluster'].map(
            lambda c: cluster_names.get(c, f"Klaster {c}") if c != -1 else "Outliers"
        )
        
        try:
            fig = px.scatter_3d(
                viz_df,
                x='x',
                y='y',
                z='z',
                color='cluster_name',
                hover_name='keyword',
                # POPRAWKA: Dodajemy cluster_name do custom_data, zawsze używamy oryginalny_wolumen jeśli kolumna istnieje
                custom_data=['oryginalny_wolumen', 'cluster_name'] if 'oryginalny_wolumen' in viz_df else ['cluster_name'],
                size=size_param, 
                size_max=25,     # <<<< ZMIANA TUTAJ: Możesz eksperymentować z tą wartością >>>>
                title='Interaktywna Mapa Klastrów Słów Kluczowych (3D)',
                labels={'x': 'Wymiar UMAP 1', 'y': 'Wymiar UMAP 2', 'z': 'Wymiar UMAP 3'},
                template='plotly_dark',
                height=700 
            )
            
            # POPRAWKA: Aktualizacja hovertemplate, aby pokazać oryginalny wolumen i prawidłową nazwę klastra
            if 'oryginalny_wolumen' in viz_df:
                fig.update_traces(
                    hovertemplate="<b>%{hovertext}</b><br><br>" +
                                  "Klaster: %{customdata[1]}<br>" + # Używamy customdata[1] dla nazwy klastra
                                  "Oryginalny Wolumen: %{customdata[0]}<extra></extra>"
                )
            else:
                 fig.update_traces(
                    hovertemplate="<b>%{hovertext}</b><br><br>" +
                                  "Klaster: %{customdata[0]}<extra></extra>" # Jeśli nie ma wolumenu, customdata[0] to cluster_name
                )


            # Dostosowanie markerów - sizeref może wymagać eksperymentów
            # Jeśli używasz 'volume_log_scaled' bezpośrednio, sizeref musi być odpowiednio mały.
            # Jeśli znormalizowałeś do 'volume_display_size' (np. 1-20), sizeref może być większy.
            sizeref_value = 0.1 
            if size_param and not viz_df[size_param].empty and viz_df[size_param].max() > 0:
                 # Bardziej dynamiczne ustawienie sizeref, jeśli jest duża rozpiętość
                 # Możesz potrzebować dostosować dzielnik (np. 40.0**2)
                 # sizeref_value = 2. * viz_df[size_param].max() / (30.**2) 
                 pass # Pozostawmy prostsze sizeref na razie, lub dostosuj jak wyżej

            fig.update_traces(
                marker=dict(
                    opacity=0.7, # Nieco mniejsza przezroczystość dla lepszej widoczności w gęstych obszarach
                    # sizeref=sizeref_value, # Eksperymentuj z tym, jeśli size_max nie wystarcza
                    # sizemode='diameter'
                )
            )
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                legend_title_text='Klaster',
                scene=dict(
                    xaxis_title='Wymiar 1',
                    yaxis_title='Wymiar 2',
                    zaxis_title='Wymiar 3',
                    aspectmode='cube' # <<<< DODANE: Lepsze proporcje dla 3D >>>>
                )
            )
            
            output_path = os.path.join(self.output_dir, "cluster_map_3d.html")
            fig.write_html(output_path)
            logger.info(f"Mapa klastrów 3D zapisana do: {output_path}")
            
            return fig
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia mapy klastrów 3D: {e}")
            fig = go.Figure()
            fig.update_layout(title_text="Błąd podczas generowania mapy klastrów 3D.")
            return fig