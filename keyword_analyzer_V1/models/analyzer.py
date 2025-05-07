"""
Moduł analizujący słowa kluczowe przy użyciu modelu Gemini lub OpenRouter.
Zoptymalizowany dla wysokiej jakości analizy z pełnym kontekstem słów kluczowych.
Wersja z poprawioną priorytetyzacją dla większej granularności.
"""
import logging
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests # Do komunikacji z OpenRouter
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
from tqdm import tqdm
import time

# Import konfiguracji
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    GEMINI_API_KEY as CONFIG_GEMINI_API_KEY, 
    GEMINI_MODEL as DEFAULT_GEMINI_MODEL, 
    VOLUME_WEIGHT as CONFIG_VOLUME_WEIGHT, 
    KD_WEIGHT as CONFIG_KD_WEIGHT, 
    CPC_WEIGHT as CONFIG_CPC_WEIGHT,
    MODEL_PRIORITY_WEIGHT as CONFIG_MODEL_PRIORITY_WEIGHT,
    LOW_PRIORITY_THRESHOLD as CONFIG_LOW_PRIORITY_THRESHOLD,
    MEDIUM_PRIORITY_THRESHOLD as CONFIG_MEDIUM_PRIORITY_THRESHOLD,
    MAX_EXPECTED_VOLUME_FOR_SCALING as CONFIG_MAX_EXPECTED_VOLUME_FOR_SCALING,
    MAX_EXPECTED_CPC_FOR_SCALING as CONFIG_MAX_EXPECTED_CPC_FOR_SCALING
)

# Próba importu stałych OpenRouter z config
# Jeśli nie istnieją, ustaw wartości domyślne
try:
    from config.config import (
        OPENROUTER_API_BASE,
        OPENROUTER_CHAT_ENDPOINT,
        OPENROUTER_MODELS_ENDPOINT,
        OPENROUTER_API_KEY as CONFIG_OPENROUTER_API_KEY
    )
except ImportError:
    # Domyślne wartości, jeśli nie zdefiniowano ich w config
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_ENDPOINT = "/chat/completions"
    OPENROUTER_MODELS_ENDPOINT = "/models"
    CONFIG_OPENROUTER_API_KEY = None
    print("UWAGA: Zmienne OpenRouter nie zostały znalezione w config.py. Używam wartości domyślnych.")

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """
    Klasa odpowiedzialna za analizę słów kluczowych z wykorzystaniem modelu Gemini lub OpenRouter.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: Optional[str] = None,
                 provider: str = "gemini"): # Dodany argument provider
        """
        Inicjalizuje analizator słów kluczowych.
        
        Args:
            api_key (Optional[str]): Klucz API dla wybranego dostawcy. 
                                     Jeśli None, spróbuje użyć klucza z config/env.
            model_name (Optional[str]): Nazwa modelu do użycia. 
                                        Jeśli None i provider='gemini', użyje domyślnego z config.
            provider (str): Dostawca modelu ('gemini' lub 'openrouter'). Domyślnie 'gemini'.
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.model = None # Obiekt modelu (dla Gemini)
        self.session = None # Sesja requests (dla OpenRouter)

        logger.info(f"Inicjalizacja KeywordAnalyzer dla dostawcy: {self.provider}")

        if self.provider == "gemini":
            self._initialize_gemini()
        elif self.provider == "openrouter":
            self._initialize_openrouter()
        else:
            raise ValueError(f"Nieznany dostawca modelu: {self.provider}. Wybierz 'gemini' lub 'openrouter'.")
            
    def _initialize_gemini(self):
        """Inicjalizuje klienta i model Gemini."""
        if not self.model_name:
            self.model_name = DEFAULT_GEMINI_MODEL # Użyj domyślnego z config
        
        if not self.api_key: # Jeśli nie podano w UI
            self.api_key = CONFIG_GEMINI_API_KEY # Spróbuj z .env przez config
        
        if not self.api_key:
            raise ValueError("Brak klucza API dla Gemini.")
            
        logger.info(f"Inicjalizacja dostawcy Gemini z modelem: {self.model_name}")
        try:
            genai.configure(api_key=self.api_key)
            # Użyj generacji config i safety settings z Twojego oryginalnego kodu
            generation_config = {"temperature": 0.2, "top_p": 0.95, "top_k": 40, "max_output_tokens": 4096}
            safety_settings = [
                {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} 
                for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                          "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
            ]
            
            try: # Logika fallback dla modelu Gemini (jak miałeś wcześniej)
                self.model = genai.GenerativeModel(model_name=self.model_name, generation_config=generation_config, safety_settings=safety_settings)
                self.model.generate_content("test connection") 
            except Exception as e:
                logger.warning(f"Problem z modelem Gemini {self.model_name}: {e}. Próbuję gemini-1.5-flash.")
                self.model_name = "gemini-1.5-flash" 
                self.model = genai.GenerativeModel(model_name=self.model_name, generation_config=generation_config, safety_settings=safety_settings)
            
            logger.info(f"Zainicjalizowano model Gemini: {self.model_name}")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji modelu Gemini: {e}")
            raise

    def _initialize_openrouter(self):
        """Inicjalizuje sesję requests dla OpenRouter."""
        if not self.model_name:
            raise ValueError("Nie podano nazwy modelu dla OpenRouter.")

        if not self.api_key: # Jeśli nie podano w UI
            self.api_key = CONFIG_OPENROUTER_API_KEY # Spróbuj z .env przez config
        
        if not self.api_key:
            raise ValueError("Brak klucza API dla OpenRouter.")

        logger.info(f"Inicjalizacja dostawcy OpenRouter z modelem: {self.model_name}")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
            # "HTTP-Referer": $YOUR_SITE_URL, # Opcjonalnie, zgodnie z dok. OpenRouter
            # "X-Title": $YOUR_SITE_NAME,   # Opcjonalnie
        })
        logger.info("Zainicjalizowano sesję dla OpenRouter.")
        # Opcjonalny test klucza
        try:
             response = self.session.get(OPENROUTER_API_BASE + OPENROUTER_MODELS_ENDPOINT, timeout=10)
             response.raise_for_status()
             logger.info("Klucz API OpenRouter wydaje się poprawny.")
        except requests.exceptions.RequestException as e:
             logger.error(f"Błąd podczas testowania klucza OpenRouter: {e}")
             # Można rzucić wyjątek, jeśli weryfikacja klucza jest krytyczna
             # raise ValueError(f"Błąd połączenia z OpenRouter lub nieprawidłowy klucz API: {e}")
             
    def _call_llm_api(self, prompt: str) -> str:
        """
        Wywołuje API wybranego dostawcy (Gemini lub OpenRouter) i zwraca odpowiedź tekstową.

        Args:
            prompt (str): Pełny tekst promptu do wysłania.

        Returns:
            str: Odpowiedź tekstowa z modelu LLM lub pusty string w przypadku błędu.
        """
        response_text = ""
        api_provider_info = f"{self.provider}/{self.model_name}" # Do logowania

        try:
            if self.provider == "gemini":
                if not self.model: raise RuntimeError("Model Gemini nie został zainicjalizowany.")
                # Dla Gemini, bezpośrednie wywołanie biblioteki
                api_response = self.model.generate_content(prompt)
                response_text = api_response.text.strip()
                logger.debug(f"Odpowiedź z Gemini ({self.model_name}) otrzymana.")
            
            elif self.provider == "openrouter":
                if not self.session: raise RuntimeError("Sesja OpenRouter nie została zainicjalizowana.")
                
                # Formatowanie dla OpenRouter (kompatybilne z OpenAI Chat API)
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2, # Można pobierać z config
                    "top_p": 0.95,      # Można pobierać z config
                    "max_tokens": 4096  # Dostosuj w razie potrzeby
                }
                api_url = OPENROUTER_API_BASE + OPENROUTER_CHAT_ENDPOINT
                
                logger.debug(f"Wysyłanie zapytania do OpenRouter ({self.model_name})...")
                api_response = self.session.post(api_url, json=payload, timeout=120) # Dłuższy timeout dla LLM
                api_response.raise_for_status() 
                
                response_data = api_response.json()
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
                    logger.debug(f"Odpowiedź z OpenRouter ({self.model_name}) otrzymana.")
                else:
                    logger.warning(f"Otrzymano nieoczekiwaną odpowiedź z OpenRouter: {response_data}")
            else:
                 # Ten warunek nie powinien wystąpić dzięki walidacji w __init__
                 raise ValueError(f"Nieobsługiwany dostawca w _call_llm_api: {self.provider}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd połączenia podczas wywołania API ({api_provider_info}): {e}")
        except genai.types.generation_types.StopCandidateException as e: # Specyficzny błąd Gemini
            logger.error(f"Generowanie przez Gemini ({self.model_name}) zatrzymane z powodu bezpieczeństwa lub innego: {e}")
            response_text = f"Błąd: Generowanie zatrzymane przez model ({e})" # Zwróć informację o błędzie
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas wywołania API LLM ({api_provider_info}): {e}")
            # logger.debug(f"Prompt powodujący błąd: {prompt[:500]}...") # Opcjonalnie do debugowania

        return response_text
    
    def name_clusters(self, df: pd.DataFrame, max_keywords_per_cluster: Optional[int] = None,
                      progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[int, str]:
        """
        Nazywa klastry na podstawie zawartych w nich słów kluczowych.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            max_keywords_per_cluster: Maksymalna liczba słów kluczowych do przesłania do API (None = wszystkie)
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Słownik mapujący ID klastrów na ich nazwy
        """
        if 'cluster' not in df.columns or 'keyword' not in df.columns:
            raise ValueError("DataFrame musi zawierać kolumny 'cluster' i 'keyword'")
        
        logger.info("Rozpoczynam nazywanie klastrów...")
        
        if progress_callback:
            progress_callback(0.0, "Przygotowanie do nazywania klastrów...")
        
        # Słownik na wyniki
        cluster_names = {}
        
        # Pobierz unikalne ID klastrów (bez -1, który oznacza szum)
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        total_clusters = len(unique_clusters)
        
        # Wybór między tqdm a progress_callback
        if progress_callback:
            iterator = enumerate(unique_clusters)
        else:
            iterator = enumerate(tqdm(unique_clusters, desc="Nazywanie klastrów"))
        
        for idx, cluster_id in iterator:
            # Pobierz słowa kluczowe z klastra
            cluster_keywords = df[df['cluster'] == cluster_id]['keyword'].tolist()
            
            # Sortowanie według wolumenu (jeśli dostępny)
            if 'volume' in df.columns:
                cluster_data = df[df['cluster'] == cluster_id].sort_values(by='volume', ascending=False)
                cluster_keywords = cluster_data['keyword'].tolist()
            
            # Ogranicz liczbę słów kluczowych tylko jeśli podano limit i jest przekroczony
            if max_keywords_per_cluster is not None and len(cluster_keywords) > max_keywords_per_cluster:
                sample_keywords = cluster_keywords[:max_keywords_per_cluster]
                keyword_info = f"Wybraną próbkę {max_keywords_per_cluster} z {len(cluster_keywords)} słów kluczowych"
            else:
                sample_keywords = cluster_keywords
                keyword_info = f"Wszystkie {len(cluster_keywords)} słów kluczowych"
            
            # Przygotuj prompt dla modelu z głębszą analizą
            prompt = f"""
            Poniżej znajduje się lista powiązanych słów kluczowych, które zostały algorytmicznie zgrupowane:
            
            {', '.join(sample_keywords)}
            
            Kontekst: Ta lista zawiera {keyword_info} w tym klastrze.
            
            Na podstawie tej listy:
            1. Zidentyfikuj wspólny temat lub dziedzinę łączącą te słowa kluczowe.
            2. Przeanalizuj intencję użytkownika stojącą za tymi słowami (informacyjna, transakcyjna, itp.).
            3. Utwórz krótką (2-5 słów), opisową nazwę dla tego klastra tematycznego.
            4. Nazwa powinna być konkretna i precyzyjna, pozwalająca odróżnić ten klaster od innych.
            5. **Kluczowe Wymaganie Językowe:** Nazwa klastra MUSI być w tym samym języku, co większość podanych słów kluczowych. 
            Jeśli słowa są po angielsku, nazwa musi być po angielsku. Jeśli po polsku, nazwa po polsku. 
            NIE TŁUMACZ nazwy na inny język, zwłaszcza na polski, jeśli oryginalne słowa są w innym języku.

            PRZYKŁAD (jeśli słowa kluczowe to "used cars for sale", "buy second-hand auto", "pre-owned vehicles"):
            Poprawna nazwa (angielska): Used Car Purchase
            Błędna nazwa (polska): Zakup Używanych Samochodów

            PRZYKŁAD (jeśli słowa kluczowe to "przepisy na ciasta domowe", "jak upiec sernik", "łatwe ciasto"):
            Poprawna nazwa (polska): Domowe Wypieki Ciast
            Błędna nazwa (angielska): Homemade Cake Recipes
            
            WAŻNE: Najpierw określ, w jakim języku są słowa kluczowe, i utwórz nazwę klastra W TYM SAMYM JĘZYKU. 
            Nie tłumacz słów kluczowych ani nazwy klastra na inny język. Zachowaj oryginalny język słów kluczowych.
            
            Odpowiedz TYLKO nazwą klastra, bez dodatkowego tekstu czy wyjaśnień.
            """
            
            try:
                # <<<< ZMIANA: Użycie nowej metody do wywołania API >>>>
                cluster_name_text = self._call_llm_api(prompt) 
                
                # Sprawdzenie pustej odpowiedzi i fallback
                if not cluster_name_text or "Błąd:" in cluster_name_text: 
                    logger.warning(f"Pusta lub błędna odpowiedź LLM dla nazwy klastra {cluster_id}. Odpowiedź: '{cluster_name_text}'")
                    cluster_name_text = f"Cluster {cluster_id} (błąd nazywania)"
                
                cluster_names[cluster_id] = cluster_name_text.strip() # Dodatkowy strip dla pewności
                
                if progress_callback:
                    progress = (idx + 1) / total_clusters
                    status = f"Nazywanie klastra {idx + 1}/{total_clusters}: {cluster_name_text}"
                    progress_callback(progress, status)
                time.sleep(1) # Dostosuj pauzę wg potrzeb

            except Exception as e: # Ten except łapie teraz błędy z logiki pętli
                logger.error(f"Nieoczekiwany błąd w pętli nazywania klastra {cluster_id}: {e}")
                cluster_names[cluster_id] = f"Cluster {cluster_id} (błąd pętli)"
        
        logger.info(f"Zakończono nazywanie {len(cluster_names)} klastrów")
        
        if progress_callback:
            progress_callback(1.0, f"Zakończono nazywanie klastrów ({len(cluster_names)} klastrów)")
        
        return cluster_names
    
    def analyze_cluster_content(self, df: pd.DataFrame, cluster_id: int, cluster_name: str, max_keywords: Optional[int] = None) -> Dict[str, Any]:
        """
        Analizuje zawartość klastra, generując wnioski i rekomendacje.
        Zawiera zmodyfikowaną logikę obliczania priorytetu.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            cluster_id: ID klastra do analizy
            cluster_name: Nazwa klastra
            max_keywords: Maksymalna liczba słów kluczowych do przesłania do API (None = wszystkie)
            
        Returns:
            Słownik zawierający wyniki analizy
        """
        # Pobierz słowa kluczowe z klastra
        cluster_df = df[df['cluster'] == cluster_id]
        
        if len(cluster_df) == 0:
            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "keywords_count": 0,
                "total_volume": 0,
                "avg_difficulty": 0, # Ustawiono domyślną trudność
                "avg_cpc": 0,        # Ustawiono domyślny CPC
                "intent_distribution": {},
                "insights": "Brak słów kluczowych w klastrze",
                "content_strategy": "Nie dotyczy",
                "priority_justification": "Brak danych",
                "priority_score": 0,
                "priority_level": "Niski"
            }
        
        # Pobierz podstawowe statystyki
        keywords_count = len(cluster_df)
        # Użyj .get(col, 0) aby uniknąć błędów przy braku kolumn
        total_volume = int(cluster_df.get('volume', 0).sum())
        avg_difficulty = round(float(cluster_df.get('difficulty', 0).mean()), 2)
        avg_cpc = round(float(cluster_df.get('cpc', 0).mean()), 2)
        
        # Określ intencje w klastrze - sprawdź nowy format intencji z Ahrefs
        intent_columns = ['branded', 'local', 'navigational', 'informational', 'commercial', 'transactional', 'unknown'] # Dodano unknown
        available_intent_columns = [col for col in intent_columns if col in cluster_df.columns]
        
        intent_distribution = {}
        intent_summary = "Brak danych o intencji"
        
        if available_intent_columns:
            # Nowy format intencji z kolumn Ahrefs
            total_keywords_with_intents = len(cluster_df) # Zakładamy, że każda fraza ma jakąś intencję
            
            for intent in available_intent_columns:
                # Upewnij się, że kolumna istnieje przed próbą sumowania
                if intent in cluster_df:
                    count = cluster_df[intent].sum()  # Suma wartości True
                    # Unikaj dzielenia przez zero
                    percentage = (count / total_keywords_with_intents) * 100 if total_keywords_with_intents > 0 else 0
                    intent_distribution[intent] = {
                        'count': int(count),
                        'percentage': round(percentage, 2)
                    }
                else:
                     intent_distribution[intent] = {'count': 0, 'percentage': 0}
            
            # Podsumowanie intencji dla prompta
            intent_summary = ", ".join([f"{intent.capitalize()}: {data['count']} ({data['percentage']}%)" 
                                       for intent, data in intent_distribution.items() if data['count'] > 0]) # Pokaż tylko istniejące intencje
            
            # Sprawdź, ile słów ma przypisaną więcej niż jedną intencję
            if total_keywords_with_intents > 0:
                multiple_intents_count = cluster_df.apply(
                    lambda row: sum(row[col] for col in available_intent_columns if col in row and row[col]), # Sprawdź czy kolumna istnieje i czy wartość jest True
                    axis=1
                )
                
                keywords_with_multiple_intents = (multiple_intents_count > 1).sum()
                multiple_intents_pct = round((keywords_with_multiple_intents / total_keywords_with_intents) * 100, 2)
                
                intent_summary += f"\nUwaga: {keywords_with_multiple_intents} słów ({multiple_intents_pct}%) ma więcej niż jedną intencję"
            else:
                 intent_summary += "\nUwaga: Brak słów kluczowych do analizy intencji."

        elif 'intent' in cluster_df.columns:
            # Stary format intencji z pojedynczej kolumny 'intent'
            intent_distribution = cluster_df['intent'].value_counts().to_dict()
            intent_summary = ", ".join([f"{intent}: {count}" for intent, count in intent_distribution.items()])
        
        # Sortuj według wolumenu (najbardziej popularne na górze)
        # Sprawdź, czy kolumna 'volume' istnieje przed sortowaniem
        if 'volume' in cluster_df.columns:
            cluster_df_sorted = cluster_df.sort_values(by='volume', ascending=False)
        else:
            cluster_df_sorted = cluster_df # Brak sortowania, jeśli nie ma wolumenu
            
        
        # Używamy wszystkich słów kluczowych, chyba że podano limit
        if max_keywords is not None and len(cluster_df_sorted) > max_keywords:
            sample_keywords_df = cluster_df_sorted.head(max_keywords)
            keyword_info = f"Próbka {max_keywords} z {len(cluster_df_sorted)} słów kluczowych (sortowane wg wolumenu, jeśli dostępny)"
        else:
            sample_keywords_df = cluster_df_sorted
            keyword_info = f"Wszystkie {len(cluster_df_sorted)} słów kluczowych w klastrze"
        
        # Wybierz kolumny do wyświetlenia, sprawdzając ich istnienie
        base_display_columns = ['keyword']
        optional_columns = ['volume', 'difficulty', 'cpc']
        display_columns = base_display_columns + [col for col in optional_columns if col in sample_keywords_df.columns]
                
        # Dodaj kolumny intencji, jeśli są dostępne
        if available_intent_columns:
             display_columns.extend([col for col in available_intent_columns if col in sample_keywords_df.columns])
        elif 'intent' in sample_keywords_df.columns:
            display_columns.append('intent')
        
        # Przygotuj formatowaną tabelę słów kluczowych (tylko z istniejących kolumn)
        keywords_table = sample_keywords_df[display_columns].to_string(index=False)
        
        # Przygotuj dodatkowe informacje statystyczne o klastrze
        stats_summary = f"""
        Statystyki klastra:
        - Liczba słów kluczowych: {keywords_count}
        - Łączny wolumen: {total_volume if 'volume' in cluster_df.columns else 'N/A'}
        - Średnia trudność (KD): {avg_difficulty if 'difficulty' in cluster_df.columns else 'N/A'}
        - Średni CPC: ${avg_cpc if 'cpc' in cluster_df.columns else 'N/A'}
        - Rozkład intencji: {intent_summary}
        """
        
        # Pobierz próbkę słów kluczowych do określenia języka
        sample_keywords_text = ", ".join(sample_keywords_df['keyword'].head(10).tolist())
        
        # Zmodyfikowany prompt z prośbą o kompleksową analizę w określonym formacie
        prompt = f"""
        # Analiza klastra słów kluczowych: "{cluster_name}"

        ## Dane klastra
        {stats_summary}

        ## Zawartość klastra
        {keyword_info}:

        {keywords_table}

        ## Zadanie
        Przeprowadź dogłębną analizę tego klastra słów kluczowych, biorąc pod uwagę wszystkie przedstawione dane. Przeanalizuj wzorce, intencje użytkowników i potencjał biznesowy.

        WAŻNE: Najpierw określ w jakim języku są słowa kluczowe (np. polskim, angielskim, niemieckim itd.). 
        Następnie:
        1. Swoją analizę napisz po polsku
        2. Wszystkie przykłady słów kluczowych, tematy, nagłówki, formaty i sugerowane frazy zachowaj w ORYGINALNYM JĘZYKU słów kluczowych (nie tłumacz ich na polski)
        3. W przypadku klastra z angielskimi słowami kluczowymi, wszystkie przykładowe tematy, tytuły, nagłówki i sugerowane frazy muszą być w języku angielskim
        4. W przypadku klastra z polskimi słowami kluczowymi, przykładowe tematy, tytuły, nagłówki i frazy powinny być w języku polskim

        Przygotuj profesjonalną analizę SEO z następującymi sekcjami, zachowując poniższą kolejność i formatowanie:

        1. PRIORITY LEVEL: 
           - Oceń priorytet tego klastra jako "High", "Medium" lub "Low" na podstawie:
             * Wolumenu wyszukiwań (czy jest znaczący?)
             * Trudności konkurencyjnej (czy jest osiągalna?)
             * Intencji komercyjnej (czy prowadzi do konwersji?)
             * Potencjału biznesowego (czy pasuje do celów?)
           - Uzasadnij swoją ocenę w 2-3 zdaniach, odnosząc się do konkretnych czynników. Bądź krytyczny - nie każdy klaster musi być High lub Medium.

        2. INSIGHTS: 
           - Krótka, zwięzła analiza (3-5 zdań) tego, co te słowa kluczowe mówią o intencjach i potrzebach użytkowników.
           - Zidentyfikuj główny temat i ewentualnie podtematy reprezentowane przez klaster.
           - Wskaż, na jakim etapie ścieżki klienta mogą znajdować się użytkownicy wpisujący te frazy.

        3. CONTENT STRATEGY: 
           - Zaproponuj ogólną strategię contentową dla tego klastra.
           - Wymień 2-3 konkretne typy treści (np. artykuł blogowy, poradnik, case study, strona produktowa), które najlepiej odpowiedziałyby na zapytania z tego klastra.
           - Dla każdego sugerowanego typu treści podaj 1-2 przykładowe tytuły lub główne tematy artykułów (w oryginalnym języku słów kluczowych).
           - Unikaj szczegółowych rekomendacji dotyczących struktury Hx, meta tagów czy wewnętrznego linkowania. Skup się na ogólnym kierunku i pomysłach na treści.

        Wykorzystaj całą swoją wiedzę o SEO, marketingu treści i zachowaniach użytkowników, aby stworzyć wartościową analizę.

        Format odpowiedzi (zachowaj dokładnie tę kolejność i nagłówki):
        PRIORITY LEVEL:
        [High/Medium/Low + Twoje uzasadnienie po polsku]

        INSIGHTS:
        [Twoja analiza po polsku, ale z zachowaniem oryginalnych słów kluczowych w przykładach]

        CONTENT STRATEGY:
        [Twoje rekomendacje po polsku, z przykładami tytułów/tematów w oryginalnym języku słów kluczowych]
        """
        
        try:
            # <<<< ZMIANA: Użycie nowej metody do wywołania API >>>>
            analysis_text = self._call_llm_api(prompt) 
            
            # Sprawdzenie pustej odpowiedzi lub błędu zwróconego przez _call_llm_api
            if not analysis_text or "Błąd:" in analysis_text:
                # W przypadku błędu API lub pustej odpowiedzi, obsłuż go tutaj
                # Zamiast rzucać wyjątek, można np. ustawić domyślne wartości
                # i kontynuować z obliczeniem priorytetu tylko na podstawie metryk
                logger.error(f"Otrzymano pustą lub błędną odpowiedź LLM podczas analizy klastra {cluster_id} ('{cluster_name}'). Odpowiedź: '{analysis_text}'")
                # Użyjemy istniejącej logiki fallback z bloku except poniżej
                raise ValueError(f"Brak poprawnej odpowiedzi z LLM: {analysis_text}") # Rzuć wyjątek, aby przejść do istniejącego bloku except
            
            # --- PARSOWANIE ODPOWIEDZI MODELU ---
            model_priority_level = "Medium"  # Domyślna wartość
            priority_justification = "Nie udało się sparsować uzasadnienia priorytetu."
            insights = "Nie udało się sparsować sekcji Insights."
            content_strategy = "Nie udało się sparsować sekcji Content Strategy."
            
            try:
                # Definiujemy znaczniki sekcji
                priority_marker = "PRIORITY LEVEL:"
                insights_marker = "INSIGHTS:"
                content_marker = "CONTENT STRATEGY:"

                # Znajdujemy pozycje znaczników w tekście odpowiedzi
                priority_start_idx = analysis_text.find(priority_marker)
                insights_start_idx = analysis_text.find(insights_marker)
                content_start_idx = analysis_text.find(content_marker)

                # Tworzymy listę znalezionych sekcji (nazwa, pozycja startowa, długość znacznika)
                # aby przetwarzać je w kolejności występowania w tekście
                sections_found = []
                if priority_start_idx != -1:
                    sections_found.append({'name': 'priority', 'start': priority_start_idx, 'marker_len': len(priority_marker)})
                if insights_start_idx != -1:
                    sections_found.append({'name': 'insights', 'start': insights_start_idx, 'marker_len': len(insights_marker)})
                if content_start_idx != -1:
                    sections_found.append({'name': 'content', 'start': content_start_idx, 'marker_len': len(content_marker)})

                # Sortujemy znalezione sekcje według ich pozycji startowej
                sections_found.sort(key=lambda x: x['start'])

                # Przetwarzamy sekcje w kolejności ich występowania
                num_found_sections = len(sections_found)
                for i, sec_info in enumerate(sections_found):
                    content_start_pos = sec_info['start'] + sec_info['marker_len']
                    
                    # Określamy koniec bieżącej sekcji:
                    # to początek następnej znalezionej sekcji lub koniec całego tekstu
                    content_end_pos = sections_found[i+1]['start'] if (i + 1) < num_found_sections else len(analysis_text)
                    
                    section_text = analysis_text[content_start_pos:content_end_pos].strip()

                    if sec_info['name'] == 'priority':
                        parts = section_text.split(maxsplit=1)
                        if parts: # Upewniamy się, że parts nie jest puste
                            level_text = parts[0].lower()
                            if "high" in level_text: 
                                model_priority_level = "High"
                            elif "low" in level_text: 
                                model_priority_level = "Low"
                            # Domyślnie pozostaje "Medium", jeśli nie znajdzie "high" ani "low" explicitly
                        
                        priority_justification = parts[1] if len(parts) > 1 else "Brak uzasadnienia od modelu lub nie udało się go wyodrębnić."
                    
                    elif sec_info['name'] == 'insights':
                        insights = section_text
                    
                    elif sec_info['name'] == 'content':
                        content_strategy = section_text
            
            except Exception as parse_error:
                 logger.error(f"Błąd podczas parsowania odpowiedzi modelu dla klastra {cluster_id} ('{cluster_name}'): {parse_error}")
                 # Jeśli wystąpił błąd, domyślne wartości "Nie udało się sparsować..." pozostaną.
                 # Można tu dodać bardziej szczegółową logikę fallback, jeśli potrzeba.
                 # Np. jeśli parsowanie całkowicie zawiedzie, można przypisać całą odpowiedź do jednej zmiennej:
                 if all("Nie udało się sparsować" in val for val in [priority_justification, insights, content_strategy]):
                     insights = f"Krytyczny błąd parsowania. Surowa odpowiedź: {analysis_text}"

            # --- OBLICZANIE PRIORYTETU ---
            
            # Oblicz priorytet na podstawie metryk (nowa funkcja z zakresem ~[0, 2])
            priority_metrics = self._calculate_priority_score_v2(
                total_volume, 
                avg_difficulty, 
                avg_cpc, 
                intent_distribution
            )
            
            # Mapowanie poziomu priorytetu z modelu na wartość liczbową [0, 0.5, 1.0]
            model_priority_mapping = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
            model_priority_score = model_priority_mapping.get(model_priority_level, 0.5) # Domyślnie 0.5 (Medium)
            
            # Łączny wynik priorytetu - ważona średnia
            # Dzielnik sumuje wagi (1 dla metryk + waga modelu)
            combined_priority = (priority_metrics + model_priority_score * CONFIG_MODEL_PRIORITY_WEIGHT) / (1 + CONFIG_MODEL_PRIORITY_WEIGHT)
            
            # Określenie końcowego poziomu priorytetu na podstawie progów
            if combined_priority >= CONFIG_MEDIUM_PRIORITY_THRESHOLD:
                final_priority_level = "Wysoki"
            elif combined_priority >= CONFIG_LOW_PRIORITY_THRESHOLD:
                final_priority_level = "Średni"
            else:
                final_priority_level = "Niski"
            
            # Przygotuj wyniki analizy
            analysis_results = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "keywords_count": keywords_count,
                "total_volume": total_volume,
                "avg_difficulty": avg_difficulty,
                "avg_cpc": avg_cpc,
                "intent_distribution": intent_distribution,
                "insights": insights,
                "content_strategy": content_strategy,
                "priority_justification": priority_justification, # Uzasadnienie z modelu
                "priority_score": round(combined_priority, 2), # Końcowy, ważony wynik
                "priority_level": final_priority_level # Końcowy poziom (Niski/Średni/Wysoki)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy klastra {cluster_id}: {e}")
            # W przypadku błędu API, oblicz priorytet tylko na podstawie metryk
            priority_metrics = self._calculate_priority_score_v2(total_volume, avg_difficulty, avg_cpc, intent_distribution)
             # Użyj tylko wyniku metrycznego do określenia poziomu (bez wpływu modelu)
            if priority_metrics >= CONFIG_MEDIUM_PRIORITY_THRESHOLD: # Użyj tych samych progów dla porównania
                fallback_priority_level = "Wysoki"
            elif priority_metrics >= CONFIG_LOW_PRIORITY_THRESHOLD:
                fallback_priority_level = "Średni"
            else:
                fallback_priority_level = "Niski"

            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "keywords_count": keywords_count,
                "total_volume": total_volume,
                "avg_difficulty": avg_difficulty,
                "avg_cpc": avg_cpc,
                "intent_distribution": intent_distribution,
                "insights": f"Wystąpił błąd podczas analizy klastra przez API: {e}",
                "content_strategy": "Nie można wygenerować rekomendacji z powodu błędu API.",
                "priority_justification": "Błąd API, priorytet oparty tylko na metrykach.",
                "priority_score": round(priority_metrics, 2), # Zwróć tylko wynik metryczny
                "priority_level": fallback_priority_level # Poziom oparty tylko na metrykach
            }
    
    def _calculate_priority_score_v2(self, volume: int, difficulty: float, cpc: float, intent_distribution: Dict[str, Any]) -> float:
        """
        Oblicza wynik priorytetu dla klastra na podstawie metryk (wersja 2).
        Normalizuje metryki do skali [0, 1] i modyfikuje wagę intencji.
        Zwraca wynik, który nie jest już ograniczony do [1, 3].
        
        Args:
            volume: Łączny wolumen wyszukiwań
            difficulty: Średnia trudność słów kluczowych (zakładane 0-100)
            cpc: Średni koszt kliknięcia
            intent_distribution: Rozkład intencji w klastrze
            
        Returns:
            Wynik priorytetu oparty na metrykach (potencjalnie poza zakresem [0, 1])
        """
        # --- Normalizacja metryk do skali [0, 1] ---
        
        # Normalizacja wolumenu (logarytmicznie, skalowana do MAX_EXPECTED_VOLUME_FOR_SCALING)
        # Funkcja logarytmiczna rosnąca od 0 do 1
        log_volume = np.log10(max(1, volume) + 1) # +1 aby uniknąć log10(1)=0, max(1,..) dla volume=0
        log_max_volume = np.log10(CONFIG_MAX_EXPECTED_VOLUME_FOR_SCALING + 1)
        volume_score = min(1.0, log_volume / log_max_volume) if log_max_volume > 0 else 0.0
        
        # Normalizacja trudności (liniowo, odwrotnie proporcjonalna)
        # Wynik 1 dla trudności 0, wynik 0 dla trudności 100
        difficulty_score = max(0.0, (100.0 - difficulty) / 100.0)
        
        # Normalizacja CPC (liniowo, skalowana do MAX_EXPECTED_CPC_FOR_SCALING)
        # Wynik 0 dla CPC 0, wynik 1 dla CPC >= MAX_EXPECTED_CPC_FOR_SCALING
        cpc_score = min(1.0, max(0.0, cpc / CONFIG_MAX_EXPECTED_CPC_FOR_SCALING))
        
        # --- Modyfikacja Wagi Intencji ---
        intent_weight = 1.0  # Bazowa waga
        bonus_weight = 0.0   # Bonus za intencje komercyjne/transakcyjne
        penalty_weight = 0.0 # Kara za intencje informacyjne/inne
        
        # Sprawdź format intencji
        is_new_format = isinstance(next(iter(intent_distribution.values()), None), dict)
        
        if is_new_format:
            total_percentage = sum(data.get('percentage', 0) for data in intent_distribution.values())
            # Unikaj dzielenia przez zero, jeśli suma procentów jest 0
            if total_percentage > 0:
                 # Oblicz udział procentowy poszczególnych intencji
                commercial_pct = intent_distribution.get('commercial', {}).get('percentage', 0) / total_percentage
                transactional_pct = intent_distribution.get('transactional', {}).get('percentage', 0) / total_percentage
                branded_pct = intent_distribution.get('branded', {}).get('percentage', 0) / total_percentage
                informational_pct = intent_distribution.get('informational', {}).get('percentage', 0) / total_percentage
                # Można dodać obsługę 'local', 'navigational', 'unknown' jeśli chcemy je karać/nagradzać

                # Bonus za intencje "wartościowe" (większa waga dla transakcyjnych)
                bonus_weight = 1.0 * transactional_pct + 0.75 * commercial_pct + 0.25 * branded_pct
                
                # Kara za intencje informacyjne (można dostosować współczynnik 0.5)
                penalty_weight = 0.2 * informational_pct 
                
                # Oblicz końcową wagę intencji
                intent_weight = (1.0 + bonus_weight) * (1.0 - penalty_weight)
            else:
                intent_weight = 1.0 # Brak danych o intencjach, waga neutralna

        else: # Stary format (value_counts)
            total_keywords = sum(intent_distribution.values())
            if total_keywords > 0:
                commercial_pct = intent_distribution.get('commercial', 0) / total_keywords
                transactional_pct = intent_distribution.get('transactional', 0) / total_keywords
                branded_pct = intent_distribution.get('branded', 0) / total_keywords # Załóżmy istnienie
                informational_pct = intent_distribution.get('informational', 0) / total_keywords

                bonus_weight = 2.0 * transactional_pct + 1.0 * commercial_pct + 0.5 * branded_pct
                penalty_weight = 0.5 * informational_pct
                intent_weight = (1.0 + bonus_weight) * (1.0 - penalty_weight)
            else:
                 intent_weight = 1.0 # Brak danych o intencjach
                 
        # Ogranicz wagę intencji, aby uniknąć ekstremalnych wartości
        intent_weight = max(0.1, min(2.5, intent_weight)) # Np. zakres [0.1, 2.5]

        # Łączny wynik priorytetu oparty na metrykach
        # Jest to suma ważona wyników [0, 1] pomnożona przez wagę intencji
        priority_score = (
            volume_score * CONFIG_VOLUME_WEIGHT +
            difficulty_score * CONFIG_KD_WEIGHT +
            cpc_score * CONFIG_CPC_WEIGHT
        ) * intent_weight
        
        # Nie stosujemy już clampingu min(3, max(1, ...))
        # Zwracamy surowy wynik ważony
        return priority_score
    
    def process_all_clusters(self, df: pd.DataFrame, cluster_names: Dict[int, str], max_keywords: Optional[int] = None,
                             progress_callback: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
        """
        Przetwarza wszystkie klastry, generując analizę dla każdego z nich.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            cluster_names: Słownik mapujący ID klastrów na ich nazwy
            max_keywords: Maksymalna liczba słów kluczowych do analizy dla każdego klastra (None = wszystkie)
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Lista słowników zawierających wyniki analizy dla każdego klastra
        """
        logger.info("Rozpoczynam analizę wszystkich klastrów...")
        
        # Lista na wyniki analizy
        all_analyses = []
        
        # Pobierz unikalne ID klastrów (bez -1, który oznacza szum)
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        total_clusters = len(unique_clusters)
        
        # Wybór między tqdm a progress_callback
        if progress_callback:
            iterator = enumerate(unique_clusters)
        else:
            iterator = enumerate(tqdm(unique_clusters, desc="Analizowanie klastrów"))
        
        for idx, cluster_id in iterator:
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            
            # Analizuj klaster (używa nowej logiki priorytetyzacji)
            analysis = self.analyze_cluster_content(df, cluster_id, cluster_name, max_keywords)
            
            # Dodaj analizę do wyników
            all_analyses.append(analysis)
            
            # Aktualizuj pasek postępu
            if progress_callback:
                progress = (idx + 1) / total_clusters
                status = f"Analiza klastra {idx + 1}/{total_clusters}: {cluster_name} -> {analysis.get('priority_level', 'N/A')}" # Dodano poziom priorytetu do statusu
                progress_callback(progress, status)
            
            # Krótka przerwa, aby uniknąć limitów API (można dostosować)
            time.sleep(1) # Zwiększono pauzę dla stabilności API przy dłuższych analizach
        
        # Sortuj klastry według priorytetu (malejąco) - używa nowego 'priority_score'
        all_analyses = sorted(all_analyses, key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"Zakończono analizę {len(all_analyses)} klastrów")
        
        if progress_callback:
            progress_callback(1.0, f"Analiza klastrów zakończona ({len(all_analyses)} klastrów)")
        
        return all_analyses