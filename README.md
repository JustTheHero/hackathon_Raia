# Ferramenta de Moderação de Fake News para Redes Sociais

## Instalar dependências
`pip install -r requirements.txt`
## Rodar API
`python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000`
## Rodar o site
```
cd site/raia
npm install
npm run dev
```

## Como funciona?
### Dados
Utilizamos a API atproto da bluesky para obtermos postagens reais, além de obtermos informações relevantes sobre o autor e os posts, que incluem: Quando a conta foi criada, qual o ratio de seguidores/seguindo e quais posts foram postados por mais de um usuário, indicando coordenação.

### Modelo
Utilizamos o fake-br-corpus, que contém notícias rotuladas como verdadeiras ou falsas. Os textos passam por pré-processamento (remoção de stopwords, normalização, vetorização por TF-IDF ou embeddings). O treinamento é feito com cross validation: dividimos os dados em k partes, treinamos em k-1 e validamos na parte restante, repetindo até usar todas as partições. Assim, o modelo aprende padrões linguísticos que distinguem fake news de notícias reais e gera métricas médias (acurácia, F1, etc.) que indicam seu desempenho.

### Checagem de fatos
A LLM atua como um terceiro pilar, complementando os sinais de usuários e padrões textuais. Ela recebe o conteúdo suspeito e aplica checagem de fatos automática, além de receber as informações do modelo anterior e os metadados da conta. Assim, a decisão final é mais robusta, e a chance de alucinação é menor.