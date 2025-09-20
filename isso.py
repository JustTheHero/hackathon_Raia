import pandas as pd
import numpy as np
import pickle
import re
import openai
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import RSLPStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

import textstat
from collections import Counter
import spacy
from spellchecker import SpellChecker
import warnings
warnings.filterwarnings('ignore')

nltk_downloads = [
    'punkt', 'stopwords', 'vader_lexicon', 'averaged_perceptron_tagger',
    'rslp', 'punkt_tab', 'mac_morpho', 'floresta', 'averaged_perceptron_tagger_eng'
]

for resource in nltk_downloads:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

class GPTAnalysisEnhancer:
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        elif os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            print("⚠️ API Key do OpenAI não fornecida. Funcionalidade GPT desabilitada.")
            self.enabled = False
            return
        
        self.enabled = True
        self.client = openai.OpenAI(api_key=openai.api_key)
    
    def enhance_analysis(self, text: str, analysis_results: Dict) -> Dict[str, str]:
        if not self.enabled:
            return {
                'gpt_analysis': 'GPT analysis não disponível - API key não configurada',
                'coherence_score': 'N/A',
                'final_recommendation': analysis_results.get('prediction', 'UNKNOWN')
            }
        
        try:
            prompt = self._create_analysis_prompt(text, analysis_results)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise de fake news em português brasileiro. Analise os dados fornecidos e forneça uma avaliação coerente e fundamentada."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            gpt_response = response.choices[0].message.content
            return self._parse_gpt_response(gpt_response, analysis_results)
            
        except Exception as e:
            return {
                'gpt_analysis': f'Erro na análise GPT: {str(e)}',
                'coherence_score': 'ERROR',
                'final_recommendation': analysis_results.get('prediction', 'UNKNOWN')
            }
    
    def _create_analysis_prompt(self, text: str, analysis: Dict) -> str:
        return f"""
Analise o seguinte texto e os resultados da análise automatizada para fake news:

TEXTO ANALISADO:
{text[:500]}...

RESULTADOS DA ANÁLISE AUTOMATIZADA:
- Predição: {analysis.get('prediction', 'N/A')}
- Confiança: {analysis.get('confidence', 0):.1%}
- Probabilidade FAKE: {analysis.get('probability_fake', 0):.1%}
- Score de Confiabilidade: {analysis.get('reliability_score', 0):.1%}

INDICADORES DETECTADOS:
- Indicadores de Fake News: {analysis.get('detailed_analysis', {}).get('fake_indicators_total', 0)}
- Emotividade: {analysis.get('detailed_analysis', {}).get('emotiveness', 0)}
- Linguagem Conspirativa: {analysis.get('detailed_analysis', {}).get('conspiracy_language', 0)}
- Pressão Temporal: {analysis.get('detailed_analysis', {}).get('time_pressure', 0)}
- Autoridade Falsa: {analysis.get('detailed_analysis', {}).get('false_authority', 0)}

QUALIDADE DO TEXTO:
- Legibilidade: {analysis.get('text_quality', {}).get('readability', 0):.1f}
- Diversidade Lexical: {analysis.get('text_quality', {}).get('lexical_diversity', 0):.3f}
- Qualidade Ortográfica: {analysis.get('text_quality', {}).get('spelling_quality', 0):.1%}

Com base nesta análise, forneça:
1. Uma avaliação coerente da classificação (CONCORDO/DISCORDO e por quê)
2. Um score de coerência de 1-10 baseado na consistência dos indicadores
3. Uma recomendação final (FAKE/TRUE/INCERTO)
4. Principais pontos que sustentam sua conclusão

Formato de resposta:
AVALIAÇÃO: [sua avaliação]
COERÊNCIA: [1-10]
RECOMENDAÇÃO: [FAKE/TRUE/INCERTO]
JUSTIFICATIVA: [principais pontos, com tom leve e facil de entender para um usuario em linguagem simples, sem citar o resultado da analise automatizada]
"""

    def _parse_gpt_response(self, response: str, original_analysis: Dict) -> Dict[str, str]:
        try:
            lines = response.strip().split('\n')
            parsed = {}
            
            for line in lines:
                if line.startswith('AVALIAÇÃO:'):
                    parsed['gpt_analysis'] = line.replace('AVALIAÇÃO:', '').strip()
                elif line.startswith('COERÊNCIA:'):
                    parsed['coherence_score'] = line.replace('COERÊNCIA:', '').strip()
                elif line.startswith('RECOMENDAÇÃO:'):
                    parsed['final_recommendation'] = line.replace('RECOMENDAÇÃO:', '').strip()
                elif line.startswith('JUSTIFICATIVA:'):
                    parsed['justification'] = line.replace('JUSTIFICATIVA:', '').strip()
            
            if 'gpt_analysis' not in parsed:
                parsed['gpt_analysis'] = response[:200] + "..."
            if 'coherence_score' not in parsed:
                parsed['coherence_score'] = 'N/A'
            if 'final_recommendation' not in parsed:
                parsed['final_recommendation'] = original_analysis.get('prediction', 'UNKNOWN')
                
            return parsed
            
        except Exception:
            return {
                'gpt_analysis': response[:200] + "...",
                'coherence_score': 'N/A',
                'final_recommendation': original_analysis.get('prediction', 'UNKNOWN'),
                'justification': 'Erro no parsing da resposta GPT'
            }

class PortugueseTextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('portuguese'))
        self.stemmer = RSLPStemmer()
        self.spell_checker = SpellChecker(language='pt')
        
        try:
            self.nlp = spacy.load('pt_core_news_sm')
            self.has_spacy = True
        except:
            print("⚠️ spaCy português não encontrado. Algumas features serão limitadas.")
            self.nlp = None
            self.has_spacy = False
        
        self.verb_patterns = {
            'subjunctive': [
                r'\b\w+(e|a|emos|em|am)$',
                r'\b\w+(asse|asses|asse|ássemos|assem)$',
                r'\b\w+(esse|esses|esse|êssemos|essem)$',
                r'\b\w+(isse|isses|isse|íssemos|issem)$'
            ],
            'imperative': [
                r'\b(faça|façam|vá|vão|venha|venham|tome|tomem|pare|parem)$',
                r'\b\w+(e|a|em|am)\b(?=\s*[!.])',
            ]
        }
        
        self.modal_verbs = {
            'poder', 'dever', 'querer', 'saber', 'conseguir', 'precisar',
            'ter que', 'haver de', 'costomar', 'soler'
        }
        
        self.fake_indicators = {
            'urgency': [
                'urgente', 'imediato', 'agora', 'já', 'rápido', 'corre', 'pressa',
                'não perca', 'última chance', 'antes que', 'acabando'
            ],
            'conspiracy': [
                'governo esconde', 'mídia não mostra', 'verdade oculta', 'conspiração',
                'manipulação', 'farsa', 'mentira', 'enganação', 'omite', 'censura'
            ],
            'emotion': [
                'absurdo', 'inacreditável', 'chocante', 'revoltante', 'indignação',
                'escândalo', 'indigna', 'revolta', 'terrível', 'horrível'
            ],
            'authority_false': [
                'médicos não querem que você saiba', 'especialistas escondem',
                'estudos secretos', 'pesquisa censurada', 'descoberta oculta'
            ],
            'superlatives': [
                'sempre', 'nunca', 'jamais', 'todos', 'ninguém', 'completamente',
                'totalmente', 'absolutamente', 'definitivamente'
            ],
            'fear_appeal': [
                'perigo', 'risco', 'morte', 'doença', 'contaminação', 'veneno',
                'tóxico', 'cancerígeno', 'prejudicial', 'fatal'
            ]
        }
        
        self.emotional_patterns = [
            r'[!]{2,}',
            r'[?]{2,}',
            r'[!?]+',
            r'\.{3,}',
        ]

    def extract_comprehensive_features(self, text: str, metadata: Dict = None) -> Dict[str, float]:
        if pd.isna(text) or text == '':
            return self._empty_features()
        
        text = str(text)
        features = {}
        
        if metadata:
            features.update(self._extract_metadata_features(metadata))
        
        features.update(self._extract_basic_counts(text))
        features.update(self._extract_morphosyntactic_features(text))
        features.update(self._extract_stylistic_features(text))
        features.update(self._extract_quality_features(text))
        features.update(self._extract_fake_indicators(text))
        
        return features
    
    def _extract_metadata_features(self, metadata: Dict) -> Dict[str, float]:
        features = {}
        
        if 'author' in metadata:
            author = str(metadata['author']).strip()
            features['has_author'] = 1.0 if author and author != 'nan' else 0.0
            features['author_length'] = len(author) if author else 0.0
        
        if 'link' in metadata:
            link = str(metadata['link'])
            features['has_link'] = 1.0 if link and link != 'nan' else 0.0
            features['link_length'] = len(link) if link else 0.0
            
            if link and link != 'nan':
                domain_features = self._analyze_domain(link)
                features.update(domain_features)
        
        if 'category' in metadata:
            category = str(metadata['category']).lower()
            features['has_category'] = 1.0 if category and category != 'nan' else 0.0
            
            suspicious_categories = ['política', 'saúde', 'celebridades', 'economia']
            features['suspicious_category'] = 1.0 if any(cat in category for cat in suspicious_categories) else 0.0
        
        if 'date' in metadata:
            try:
                date_obj = pd.to_datetime(metadata['date'])
                features['has_date'] = 1.0
                features['is_recent'] = 1.0 if (datetime.now() - date_obj).days < 30 else 0.0
            except:
                features['has_date'] = 0.0
                features['is_recent'] = 0.0
        
        return features
    
    def _extract_basic_counts(self, text: str) -> Dict[str, float]:
        features = {}
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_no_punct = [w for w in words if w.isalpha()]
        
        features['num_tokens'] = len(words)
        features['num_words_no_punct'] = len(words_no_punct)
        features['num_sentences'] = len(sentences)
        features['num_characters'] = len(text)
        
        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        features['num_links'] = len(links)
        
        uppercase_words = re.findall(r'\b[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ]{2,}\b', text)
        features['num_uppercase_words'] = len(uppercase_words)
        
        unique_words = set(words_no_punct)
        features['num_types'] = len(unique_words)
        features['diversity'] = len(unique_words) / max(len(words_no_punct), 1)
        
        features['avg_sentence_length'] = np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0
        features['avg_word_length'] = np.mean([len(w) for w in words_no_punct]) if words_no_punct else 0
        
        return features
    
    def _extract_morphosyntactic_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        words = word_tokenize(text.lower())
        words_no_punct = [w for w in words if w.isalpha()]
        
        if not words_no_punct:
            return {key: 0.0 for key in [
                'num_verbs', 'num_subjunctive_imperative', 'num_nouns', 
                'num_adjectives', 'num_adverbs', 'num_modal_verbs',
                'num_personal_pronouns_sg', 'num_personal_pronouns_pl',
                'num_pronouns_total', 'verb_ratio', 'noun_ratio', 'adj_ratio'
            ]}
        
        pos_tags = pos_tag(words)
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        total_words = len(words_no_punct)
        
        verb_tags = ['V', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        num_verbs = sum(pos_counts.get(tag, 0) for tag in verb_tags)
        features['num_verbs'] = num_verbs
        features['verb_ratio'] = num_verbs / max(total_words, 1)
        
        noun_tags = ['N', 'NN', 'NNS', 'NNP', 'NNPS']
        num_nouns = sum(pos_counts.get(tag, 0) for tag in noun_tags)
        features['num_nouns'] = num_nouns
        features['noun_ratio'] = num_nouns / max(total_words, 1)
        
        adj_tags = ['JJ', 'JJR', 'JJS']
        num_adjectives = sum(pos_counts.get(tag, 0) for tag in adj_tags)
        features['num_adjectives'] = num_adjectives
        features['adj_ratio'] = num_adjectives / max(total_words, 1)
        
        adv_tags = ['RB', 'RBR', 'RBS']
        num_adverbs = sum(pos_counts.get(tag, 0) for tag in adv_tags)
        features['num_adverbs'] = num_adverbs
        
        text_lower = text.lower()
        
        modal_count = sum(1 for modal in self.modal_verbs if modal in text_lower)
        features['num_modal_verbs'] = modal_count
        
        subjunctive_imperative_count = 0
        for category, patterns in self.verb_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                subjunctive_imperative_count += len(matches)
        features['num_subjunctive_imperative'] = subjunctive_imperative_count
        
        personal_pronouns_sg = ['eu', 'tu', 'ele', 'ela', 'você', 'me', 'te', 'se']
        personal_pronouns_pl = ['nós', 'vocês', 'eles', 'elas', 'nos']
        
        features['num_personal_pronouns_sg'] = sum(1 for p in personal_pronouns_sg if f' {p} ' in f' {text_lower} ')
        features['num_personal_pronouns_pl'] = sum(1 for p in personal_pronouns_pl if f' {text_lower} ' in f' {text_lower} ')
        
        all_pronouns = personal_pronouns_sg + personal_pronouns_pl + [
            'este', 'esta', 'esse', 'essa', 'aquele', 'aquela', 'isto', 'isso', 'aquilo',
            'qual', 'que', 'quem', 'onde', 'quando', 'como', 'quanto'
        ]
        features['num_pronouns_total'] = sum(1 for p in all_pronouns if f' {p} ' in f' {text_lower} ')
        
        return features
    
    def _extract_stylistic_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        features['num_exclamations'] = text.count('!')
        features['num_questions'] = text.count('?')
        features['num_ellipsis'] = text.count('...')
        
        emotional_punct_count = 0
        for pattern in self.emotional_patterns:
            matches = re.findall(pattern, text)
            emotional_punct_count += len(matches)
        features['emotional_punctuation'] = emotional_punct_count
        
        pause_indicators = [',', ';', ':', '...', ' - ', ' — ']
        features['pausality'] = sum(text.count(p) for p in pause_indicators)
        
        emotional_words = [
            'amor', 'ódio', 'raiva', 'feliz', 'triste', 'medo', 'ansioso',
            'preocupado', 'nervoso', 'calmo', 'tranquilo', 'estressado'
        ]
        emotiveness = sum(1 for word in emotional_words if word in text.lower())
        
        emotiveness += features['num_exclamations'] * 0.5
        emotiveness += features['num_questions'] * 0.3
        emotiveness += emotional_punct_count * 0.7
        
        features['emotiveness'] = emotiveness
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(text)
            features.update({
                'sentiment_positive': sentiment['pos'],
                'sentiment_negative': sentiment['neg'], 
                'sentiment_neutral': sentiment['neu'],
                'sentiment_compound': sentiment['compound']
            })
        except:
            features.update({
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0,
                'sentiment_compound': 0.0
            })
        
        return features
    
    def _extract_quality_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        words = word_tokenize(text.lower())
        words_alpha = [w for w in words if w.isalpha() and len(w) > 2]
        
        if words_alpha:
            misspelled = self.spell_checker.unknown(words_alpha)
            actual_errors = [w for w in misspelled if not w[0].isupper() and len(w) > 3]
            features['spelling_errors_ratio'] = len(actual_errors) / len(words_alpha)
            features['num_spelling_errors'] = len(actual_errors)
        else:
            features['spelling_errors_ratio'] = 0.0
            features['num_spelling_errors'] = 0.0
        
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        except:
            features['flesch_reading_ease'] = 50.0
            features['flesch_kincaid_grade'] = 8.0
        
        sentences = sent_tokenize(text)
        if sentences:
            sentence_lengths = [len(word_tokenize(s)) for s in sentences]
            features['sentence_length_variance'] = np.var(sentence_lengths)
            features['max_sentence_length'] = max(sentence_lengths)
            features['min_sentence_length'] = min(sentence_lengths)
        else:
            features['sentence_length_variance'] = 0.0
            features['max_sentence_length'] = 0.0
            features['min_sentence_length'] = 0.0
        
        return features
    
    def _extract_fake_indicators(self, text: str) -> Dict[str, float]:
        features = {}
        text_lower = text.lower()
        
        for category, indicators in self.fake_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            features[f'fake_indicator_{category}'] = count
            
            text_words = len(word_tokenize(text))
            features[f'fake_indicator_{category}_density'] = count / max(text_words, 1)
        
        total_fake_indicators = sum(features[k] for k in features.keys() if k.startswith('fake_indicator_') and not k.endswith('_density'))
        features['fake_indicators_total'] = total_fake_indicators
        
        time_pressure_patterns = [
            r'antes que.*tarde', r'última.*chance', r'só.*hoje',
            r'por.*tempo.*limitado', r'enquanto.*tempo'
        ]
        time_pressure_count = sum(len(re.findall(pattern, text_lower)) for pattern in time_pressure_patterns)
        features['time_pressure'] = time_pressure_count
        
        false_authority_patterns = [
            r'médicos.*não.*querem', r'governo.*esconde', r'mídia.*omite',
            r'especialistas.*censurados', r'pesquisa.*proibida'
        ]
        false_authority_count = sum(len(re.findall(pattern, text_lower)) for pattern in false_authority_patterns)
        features['false_authority'] = false_authority_count
        
        conspiracy_patterns = [
            r'verdade.*oculta', r'eles.*não.*querem', r'sistema.*corrupto',
            r'manipulação.*massas', r'controle.*mental'
        ]
        conspiracy_count = sum(len(re.findall(pattern, text_lower)) for pattern in conspiracy_patterns)
        features['conspiracy_language'] = conspiracy_count
        
        return features
    
    def _analyze_domain(self, url: str) -> Dict[str, float]:
        features = {}
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            features['domain_length'] = len(domain)
            features['domain_has_numbers'] = 1.0 if re.search(r'\d', domain) else 0.0
            features['domain_has_hyphens'] = 1.0 if '-' in domain else 0.0
            features['domain_subdomains'] = len(domain.split('.')) - 2
            
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
            features['suspicious_tld'] = 1.0 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0.0
            
            trusted_domains = ['g1.globo.com', 'folha.uol.com.br', 'estadao.com.br', 'bbc.com']
            features['trusted_domain'] = 1.0 if any(trusted in domain for trusted in trusted_domains) else 0.0
            
        except:
            features.update({
                'domain_length': 0.0,
                'domain_has_numbers': 0.0,
                'domain_has_hyphens': 0.0,
                'domain_subdomains': 0.0,
                'suspicious_tld': 0.0,
                'trusted_domain': 0.0
            })
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        base_features = {
            'has_author': 0.0, 'author_length': 0.0, 'has_link': 0.0, 'link_length': 0.0,
            'has_category': 0.0, 'suspicious_category': 0.0, 'has_date': 0.0, 'is_recent': 0.0,
            'num_tokens': 0.0, 'num_words_no_punct': 0.0, 'num_sentences': 0.0,
            'num_characters': 0.0, 'num_links': 0.0, 'num_uppercase_words': 0.0,
            'num_types': 0.0, 'diversity': 0.0, 'avg_sentence_length': 0.0, 'avg_word_length': 0.0,
            'num_verbs': 0.0, 'num_subjunctive_imperative': 0.0, 'num_nouns': 0.0,
            'num_adjectives': 0.0, 'num_adverbs': 0.0, 'num_modal_verbs': 0.0,
            'num_personal_pronouns_sg': 0.0, 'num_personal_pronouns_pl': 0.0,
            'num_pronouns_total': 0.0, 'verb_ratio': 0.0, 'noun_ratio': 0.0, 'adj_ratio': 0.0,
            'num_exclamations': 0.0, 'num_questions': 0.0, 'num_ellipsis': 0.0,
            'emotional_punctuation': 0.0, 'pausality': 0.0, 'emotiveness': 0.0,
            'sentiment_positive': 0.0, 'sentiment_negative': 0.0, 'sentiment_neutral': 1.0,
            'sentiment_compound': 0.0,
            'spelling_errors_ratio': 0.0, 'num_spelling_errors': 0.0,
            'flesch_reading_ease': 50.0, 'flesch_kincaid_grade': 8.0,
            'sentence_length_variance': 0.0, 'max_sentence_length': 0.0, 'min_sentence_length': 0.0,
            'domain_length': 0.0, 'domain_has_numbers': 0.0, 'domain_has_hyphens': 0.0,
            'domain_subdomains': 0.0, 'suspicious_tld': 0.0, 'trusted_domain': 0.0
        }
        
        for category in self.fake_indicators.keys():
            base_features[f'fake_indicator_{category}'] = 0.0
            base_features[f'fake_indicator_{category}_density'] = 0.0
        
        base_features.update({
            'fake_indicators_total': 0.0,
            'time_pressure': 0.0,
            'false_authority': 0.0,
            'conspiracy_language': 0.0
        })
        
        return base_features

class EnhancedFakeNewsClassifier:
    def __init__(self, openai_api_key: str = None):
        self.analyzer = PortugueseTextAnalyzer()
        self.gpt_enhancer = GPTAnalysisEnhancer(openai_api_key)
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.model = None
        self.feature_names = None
        self.training_stats = {}
        
    def train(self, df: pd.DataFrame, text_column: str, label_column: str, metadata_columns: List[str] = None) -> Dict[str, Any]:
        print("=" * 80)
        print("SISTEMA AVANÇADO DE CLASSIFICAÇÃO DE FAKE NEWS - PORTUGUÊS BRASILEIRO")
        print("=" * 80)
    
        texts = df[text_column].fillna('').astype(str)
    
        metadata_list = []
        if metadata_columns:
            for idx in df.index:
                metadata = {col: df.loc[idx, col] if col in df.columns else None 
                           for col in metadata_columns}
                metadata_list.append(metadata)
        else:
            metadata_list = [{}] * len(texts)
    
        if df[label_column].dtype == 'object':
            label_mapping = {'fake': 1, 'true': 0, 'FAKE': 1, 'TRUE': 0}
            y = df[label_column].map(label_mapping).fillna(0).astype(int)
        else:
            y = df[label_column].astype(int)

        class_distribution = pd.Series(y).value_counts()
        print(f"\nDistribuição das classes:")
        for class_val, count in class_distribution.items():
            label_name = 'FAKE' if class_val == 1 else 'TRUE'
            print(f"  {label_name}: {count:,} ({count/len(y)*100:.1f}%)")

        print(f"\nExtraindo features abrangentes de {len(texts):,} textos...")
        linguistic_features = []

        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            if i % 1000 == 0:
                print(f"  Processado: {i:,}/{len(texts):,}")
        
            features = self.analyzer.extract_comprehensive_features(text, metadata)
            linguistic_features.append(list(features.values()))

        linguistic_features = np.array(linguistic_features)

        print("Extraindo features TF-IDF...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 2),
            stop_words=list(self.analyzer.stop_words),
            lowercase=True,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()

        self.feature_scaler = StandardScaler()
        linguistic_features_scaled = self.feature_scaler.fit_transform(linguistic_features)

        X_combined = np.hstack([tfidf_features, linguistic_features_scaled])

        print(f"\nResumo das Features:")
        print(f"  Features TF-IDF: {tfidf_features.shape[1]:,}")
        print(f"  Features Linguísticas: {linguistic_features.shape[1]:,}")
        print(f"  Features Totais: {X_combined.shape[1]:,}")

        tfidf_feature_names = list(self.tfidf_vectorizer.get_feature_names_out())
        linguistic_feature_names = list(self.analyzer._empty_features().keys())
        self.feature_names = tfidf_feature_names + linguistic_feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        print(f"\nDivisão dos dados:")
        print(f"  Treino: {X_train.shape[0]:,} amostras")
        print(f"  Teste: {X_test.shape[0]:,} amostras")

        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                C=0.5,
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
                n_jobs=-1
            ),
            'LinearSVC': LinearSVC(
                C=0.8,
                class_weight='balanced',
                max_iter=3000,
                random_state=42
            )
        }

        print(f"\nValidação Cruzada (5-fold):")
        print("-" * 50)

        best_model = None
        best_score = 0
        best_model_name = ""
        results = {}

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            print(f"\n{name}:")
        
            cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
            cv_precision = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision', n_jobs=-1)
            cv_recall = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall', n_jobs=-1)
            cv_accuracy = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        
            mean_f1 = cv_f1.mean()
            std_f1 = cv_f1.std()
        
            print(f"  F1-Score:  {mean_f1:.3f} ± {std_f1:.3f}")
            print(f"  Precision: {cv_precision.mean():.3f} ± {cv_precision.std():.3f}")
            print(f"  Recall:    {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")
            print(f"  Accuracy:  {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
            test_f1 = f1_score(y_test, y_pred)
            print(f"  F1-Teste:  {test_f1:.3f}")
        
            results[name] = {
                'cv_f1_mean': mean_f1,
                'cv_f1_std': std_f1,
                'test_f1': test_f1,
                'cv_precision': cv_precision.mean(),
                'cv_recall': cv_recall.mean(),
                'cv_accuracy': cv_accuracy.mean()
            }
        
            if mean_f1 > best_score:
                best_score = mean_f1
                best_model = model
                best_model_name = name

        print(f"\n" + "=" * 50)
        print(f"MELHOR MODELO: {best_model_name}")
        print(f"F1-Score CV: {best_score:.3f}")
        print("=" * 50)

        self.model = best_model
        self.model.fit(X_combined, y)

        final_predictions = self.model.predict(X_test)
        final_report = classification_report(y_test, final_predictions, 
                                   target_names=['TRUE', 'FAKE'], 
                                   output_dict=True)

        print(f"\nRelatório de Classificação Final:")
        print(classification_report(y_test, final_predictions, target_names=['TRUE', 'FAKE']))

        cm = confusion_matrix(y_test, final_predictions)
        print(f"\nMatriz de Confusão:")
        print(f"              Predito")
        print(f"              TRUE  FAKE")
        print(f"Real TRUE  [  {cm[0,0]:4d}  {cm[0,1]:4d} ]")
        print(f"     FAKE  [  {cm[1,0]:4d}  {cm[1,1]:4d} ]")

        feature_importance = self._analyze_feature_importance()

        self.training_stats = {
            'model_name': best_model_name,
            'cv_results': results,
            'final_report': final_report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'class_distribution': dict(class_distribution),
            'total_samples': len(y)
        }

        self.save_model('enhanced_fake_news_classifier_pt.pkl')

        return self.training_stats
    
    def _analyze_feature_importance(self) -> Dict[str, float]:
        if not hasattr(self.model, 'feature_importances_') and not hasattr(self.model, 'coef_'):
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_[0])
        
        feature_importance = dict(zip(self.feature_names, importances))
        
        tfidf_importance = {}
        linguistic_importance = {}
        
        for feature, importance in feature_importance.items():
            if feature in self.analyzer._empty_features().keys():
                linguistic_importance[feature] = importance
            else:
                tfidf_importance[feature] = importance
        
        print(f"\n" + "=" * 60)
        print("ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
        print("=" * 60)
        
        print(f"\nTop 15 Features TF-IDF Mais Importantes:")
        print("-" * 40)
        top_tfidf = sorted(tfidf_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (feature, importance) in enumerate(top_tfidf, 1):
            print(f"{i:2d}. {feature}: {importance:.4f}")
        
        print(f"\nTop 15 Features Linguísticas Mais Importantes:")
        print("-" * 45)
        top_linguistic = sorted(linguistic_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (feature, importance) in enumerate(top_linguistic, 1):
            print(f"{i:2d}. {feature}: {importance:.4f}")
        
        return feature_importance
    
    def predict(self, texts: List[str], metadata_list: List[Dict] = None) -> List[Dict[str, Any]]:
        if self.model is None:
            raise ValueError("Modelo não foi treinado! Execute o método train() primeiro.")

        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        linguistic_features = []
        for text, metadata in zip(texts, metadata_list):
            features = self.analyzer.extract_comprehensive_features(text, metadata)
            linguistic_features.append(list(features.values()))

        linguistic_features = np.array(linguistic_features)

        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        linguistic_features_scaled = self.feature_scaler.transform(linguistic_features)
        X_combined = np.hstack([tfidf_features, linguistic_features_scaled])

        predictions = self.model.predict(X_combined)

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_combined)
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_combined)
            probabilities = np.column_stack([
                1 / (1 + np.exp(scores)),
                1 / (1 + np.exp(-scores))
            ])
        else:
            probabilities = np.zeros((len(predictions), 2))

        results = []
        for i, (text, metadata, pred, probs) in enumerate(zip(texts, metadata_list, predictions, probabilities)):
            individual_features = self.analyzer.extract_comprehensive_features(text, metadata)
            reliability_score = self._calculate_reliability_score(individual_features, probs)
        
            base_result = {
                'text_preview': text[:150] + '...' if len(text) > 150 else text,
                'prediction': 'FAKE' if pred == 1 else 'TRUE',
                'confidence': float(max(probs)),
                'probability_fake': float(probs[1]),
                'probability_true': float(probs[0]),
                'reliability_score': reliability_score,
            
                'detailed_analysis': {
                    'fake_indicators_total': individual_features.get('fake_indicators_total', 0),
                    'emotiveness': individual_features.get('emotiveness', 0),
                    'spelling_errors_ratio': individual_features.get('spelling_errors_ratio', 0),
                    'sentiment_compound': individual_features.get('sentiment_compound', 0),
                    'diversity': individual_features.get('diversity', 0),
                    'uppercase_words': individual_features.get('num_uppercase_words', 0),
                    'exclamations': individual_features.get('num_exclamations', 0),
                    'conspiracy_language': individual_features.get('conspiracy_language', 0),
                    'time_pressure': individual_features.get('time_pressure', 0),
                    'false_authority': individual_features.get('false_authority', 0)
                },
            
                'text_quality': {
                    'readability': individual_features.get('flesch_reading_ease', 0),
                    'avg_sentence_length': individual_features.get('avg_sentence_length', 0),
                    'lexical_diversity': individual_features.get('diversity', 0),
                    'spelling_quality': 1 - individual_features.get('spelling_errors_ratio', 0)
                }
            }
            
            gpt_analysis = self.gpt_enhancer.enhance_analysis(text, base_result)
            base_result.update(gpt_analysis)
            
            results.append(base_result)

        self.save_model('enhanced_fake_news_classifier_pt.pkl')
        return results
    
    def _calculate_reliability_score(self, features: Dict, probabilities: np.ndarray) -> float:
        reliability = 0.5
        
        confidence = max(probabilities)
        reliability += (confidence - 0.5) * 0.4
        
        spelling_quality = 1 - features.get('spelling_errors_ratio', 0)
        reliability += spelling_quality * 0.1
        
        diversity = features.get('diversity', 0)
        reliability += min(diversity, 0.5) * 0.2
        
        fake_indicators = features.get('fake_indicators_total', 0)
        if fake_indicators > 3:
            reliability -= 0.1
        
        emotiveness = features.get('emotiveness', 0)
        if emotiveness > 5:
            reliability -= 0.1
        
        return max(0.0, min(1.0, reliability))
    
    def save_model(self, filename: str):
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_scaler': self.feature_scaler,
            'analyzer': self.analyzer,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filename}")
    
    def load_model(self, filename: str):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.feature_scaler = model_data['feature_scaler']
        self.analyzer = model_data['analyzer']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data.get('training_stats', {})
        
        print(f"Modelo carregado de: {filename}")
    
    def generate_analysis_report(self, results: List[Dict]) -> str:
        total = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        true_count = total - fake_count
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_reliability = np.mean([r['reliability_score'] for r in results])
        
        report = f"""
RELATÓRIO DE ANÁLISE DE FAKE NEWS COM GPT-4 MINI
{'='*60}

RESUMO GERAL:
• Total de textos analisados: {total}
• Classificados como FAKE: {fake_count} ({fake_count/total*100:.1f}%)
• Classificados como TRUE: {true_count} ({true_count/total*100:.1f}%)
• Confiança média: {avg_confidence:.1%}
• Score de confiabilidade médio: {avg_reliability:.1%}

ANÁLISE DETALHADA:
"""
        
        for i, result in enumerate(results, 1):
            analysis = result['detailed_analysis']
            quality = result['text_quality']
            
            gpt_analysis = result.get('gpt_analysis', 'N/A')
            coherence = result.get('coherence_score', 'N/A')
            final_rec = result.get('final_recommendation', result['prediction'])
            justification = result.get('justification', 'N/A')
            
            report += f"""
--- TEXTO {i} ---
Predição ML: {result['prediction']} (Confiança: {result['confidence']:.1%})
Recomendação GPT: {final_rec}
Preview: {result['text_preview']}

ANÁLISE GPT-4 MINI:
{gpt_analysis}
Score de Coerência: {coherence}
Justificativa: {justification}

INDICADORES TÉCNICOS:
• Fake indicators: {analysis['fake_indicators_total']}
• Emotividade: {analysis['emotiveness']:.1f}
• Linguagem conspirativa: {analysis['conspiracy_language']}
• Pressão temporal: {analysis['time_pressure']}
• Autoridade falsa: {analysis['false_authority']}

QUALIDADE DO TEXTO:
• Legibilidade: {quality['readability']:.1f}
• Diversidade lexical: {quality['lexical_diversity']:.3f}
• Qualidade ortográfica: {quality['spelling_quality']:.1%}
• Comprimento médio das sentenças: {quality['avg_sentence_length']:.1f}

Score de Confiabilidade: {result['reliability_score']:.1%}
{'─'*70}
"""
        
        return report

def train_enhanced_model(csv_path: str = './sample_data/pre-processed.csv', openai_api_key: str = None):
    print("Carregando dados...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Dados carregados: {len(df):,} registros")
    except FileNotFoundError:
        print(f"Arquivo '{csv_path}' não encontrado!")
        return None
    
    required_columns = ['preprocessed_news', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Colunas obrigatórias ausentes: {missing_columns}")
        print(f"Colunas disponíveis: {list(df.columns)}")
        return None
    
    metadata_columns = []
    optional_metadata = ['author', 'link', 'category', 'date']
    for col in optional_metadata:
        if col in df.columns:
            metadata_columns.append(col)
    
    print(f"Metadados disponíveis: {metadata_columns}")
    
    classifier = EnhancedFakeNewsClassifier(openai_api_key)
    
    try:
        results = classifier.train(
            df=df,
            text_column='preprocessed_news',
            label_column='label',
            metadata_columns=metadata_columns if metadata_columns else None
        )
        
        print(f"\nTREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"Modelo salvo como: enhanced_fake_news_classifier_pt.pkl")
        
        return classifier
        
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        return None

import pandas as pd
import os

def test_enhanced_model(openai_api_key: str = None, csv_file: str = './sample_data/deusehbom1.csv'):     
    print("Carregando modelo...")     
    classifier = EnhancedFakeNewsClassifier(openai_api_key)          
    
    try:         
        classifier.load_model('enhanced_fake_news_classifier_pt.pkl')     
    except FileNotFoundError:         
        print("Modelo não encontrado! Execute train_enhanced_model() primeiro.")         
        return
    
    # Verificar se o arquivo CSV existe
    if not os.path.exists(csv_file):
        print(f"Arquivo {csv_file} não encontrado!")
        return
    
    # Carregar dados do CSV
    print(f"Carregando dados do arquivo {csv_file}...")
    
    # Lista de encodings para tentar
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    df = None
    
    for encoding in encodings:
        try:
            print(f"Tentando encoding: {encoding}...")
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"✅ Arquivo carregado com sucesso usando encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Erro com encoding {encoding}: {e}")
            continue
    
    if df is None:
        print("❌ Não foi possível carregar o arquivo com nenhum dos encodings testados.")
        print("Encodings tentados:", encodings)
        return
    
    try:
        
        # Verificar se as colunas necessárias existem
        if 'clean_text' not in df.columns:
            print("Coluna 'clean_text' não encontrada no CSV!")
            return
        
        # Remover linhas com texto vazio ou NaN
        df = df.dropna(subset=['clean_text'])
        df = df[df['clean_text'].str.strip() != '']
        
        if len(df) == 0:
            print("Nenhum texto válido encontrado no arquivo CSV!")
            return
        
        textos_teste = df['clean_text'].tolist()
        indices = df['index'].tolist() if 'index' in df.columns else list(range(1, len(textos_teste) + 1))
        
        print(f"Carregados {len(textos_teste)} textos para análise.")
        
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        return
          
    print("Analisando textos com GPT-4 Mini...")     
    results = classifier.predict(textos_teste)          
    
    print("\n" + "="*80)     
    print("RESULTADOS DA ANÁLISE COM GPT-4 MINI")     
    print("="*80)          
    
    for i, (result, idx) in enumerate(zip(results, indices)):         
        prediction = result['prediction']         
        confidence = result['confidence']         
        reliability = result['reliability_score']         
        gpt_rec = result.get('final_recommendation', 'N/A')         
        coherence = result.get('coherence_score', 'N/A')                  
        
        color = '🔴' if prediction == 'FAKE' else '🟢'         
        gpt_color = '🔴' if gpt_rec == 'FAKE' else ('🟢' if gpt_rec == 'TRUE' else '🟡')                  
        
        print(f"\n{color} TEXTO {idx} (linha {i+1}) - ML: {prediction} | {gpt_color} GPT: {gpt_rec}")         
        print(f"{'─'*60}")         
        print(f"Preview: {result['text_preview']}")         
        print(f"Confiança ML: {confidence:.1%} | Confiabilidade: {reliability:.1%}")         
        print(f"Coerência GPT: {coherence}")                  
        
        if 'gpt_analysis' in result:             
            print(f"Análise GPT: {result['gpt_analysis']}")                  
        
        if 'justification' in result:             
            print(f"Justificativa: {result['justification']}")          
    
    # Gerar relatório
    report = classifier.generate_analysis_report(results)          
    
    # Salvar relatório com informações do CSV
    report_filename = f'gpt_enhanced_analysis_report_{os.path.splitext(csv_file)[0]}.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:         
        f.write(f"RELATÓRIO DE ANÁLISE - ARQUIVO: {csv_file}\n")
        f.write(f"Total de textos analisados: {len(textos_teste)}\n")
        f.write(f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(report)          
    
    print(f"\nRelatório completo com análise GPT salvo em: {report_filename}")     
    print(f"Modelo atualizado e salvo automaticamente.")
    
    # Estatísticas resumidas
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    true_count = len(results) - fake_count
    
    print(f"\n📊 ESTATÍSTICAS RESUMIDAS:")
    print(f"   Total analisado: {len(results)} textos")
    print(f"   🔴 Fake News: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    print(f"   🟢 Notícias Verdadeiras: {true_count} ({true_count/len(results)*100:.1f}%)")
    return report_filename


def main():
    choice = input("Opção (1, 2 ou 3): ").strip()
    
    api_key = None
    if choice in ['1', '2']:
        api_key = input("Digite sua API key do OpenAI: ").strip()
        if not api_key:
            print("API key não fornecida. Continuando sem GPT...")
    
    if choice == '1':
        train_enhanced_model(openai_api_key=api_key)
    elif choice == '2':
        return test_enhanced_model(openai_api_key=api_key)
    elif choice == '3':
        subchoice = input("Treinar (1) ou Testar (2)? ").strip()
        if subchoice == '1':
            train_enhanced_model()
        elif subchoice == '2':
            test_enhanced_model()
    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()