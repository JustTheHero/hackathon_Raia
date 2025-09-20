import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

def preprocess_data(df):
    if 'text' not in df.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'text'.")

    
    df = df.reset_index(drop=True)
    df.index = df.index + 1 
    
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    result_df = pd.DataFrame({
        'index': df.index,
        'clean_text': df['clean_text'],
    })
    
    result_df = result_df[result_df['clean_text'] != ''].reset_index(drop=True)
    result_df['index'] = result_df.index + 1
    
    return result_df

def process_csv_file(file_path, output_path=None):
    try:
        df = pd.read_csv(
            file_path,
            sep=',',
            quotechar='"',
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        processed_df = preprocess_data(df)
        
        if output_path:
            processed_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ Dados processados salvos em: {output_path}")
        
        return processed_df
    
    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")
        return None

if __name__ == "__main__":
    input_file = 'hackaton_vacinaPT_20250920.csv'  
    output_file = 'output_clean.csv'
    
    processed_data = process_csv_file(input_file, output_file)
    
    if processed_data is not None:
        print(f"\nTotal de linhas processadas: {len(processed_data)}")
        print(processed_data.head(10))
