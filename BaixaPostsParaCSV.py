from dotenv import load_dotenv
import os
from atproto import Client, models
from datetime import datetime
import csv
import time
import random

def search_posts(query, limit=100, cursor=None):
    """Busca posts por termos/hashtags e retorna alguns dados"""
    params = models.AppBskyFeedSearchPosts.Params(
        q=query,
        limit=limit,
        cursor=cursor
    )
    
    try: 
        response = client.app.bsky.feed.search_posts(params)
        posts =[]
        for post in response.posts:
            # Ignoraremos posts com embeds problemáticos
            # Que podem ter tipos complexos que o pydantic não identifica
            try:
                # Obtem as hashtags do post
                hashtags = []
                if hasattr(post.record, 'facets') and post.record.facets:
                    for facet in post.record.facets:
                        if hasattr(facet, 'features'):
                            for feature in facet.features:
                                if hasattr(feature, 'tag') and feature.tag:
                                    hashtags.append(feature.tag)

                posts.append({
                    'uri': post.uri,
                    'text': post.record.text,
                    'author_handle': post.author.handle,
                    'author_display_name': post.author.display_name,
                    'created_at': post.record.created_at,
                    'like_count': post.like_count or 0,
                    'reply_count': post.reply_count or 0,
                    'repost_count': post.repost_count or 0,
                    'hashtags': ', '.join(hashtags) if hashtags else None,
                    'url': f'https://bsky.app/profile/{post.author.handle}/post/{post.uri.split("/")[-1]}'
                })
            except Exception as e:
                print(f"Erro ao processar post {post.uri}: {e}")
             
        return posts, response.cursor        
            
    except Exception as e:
        print(f"Erro na busca do post: {e}")
        return [], None

def collect_for_query(query, batch_size, max_posts, seen_uris, writer):
    cursor = None
    total_collected = 0
    print("Obtendo dados para a query atual:", query)

    while total_collected < max_posts:
        print(f"Buscando {batch_size} posts... (Total: {total_collected})")
        posts, cursor = search_posts(query, limit=batch_size, cursor=cursor)

        # Mais delay pode ajudar a baixar mais posts!
        time.sleep(1.0 + random.uniform(0, 1))

        if not posts:
            break  # Sem mais resultados
        
        # Verifica quais posts não foram salvos ainda
        new_posts = [p for p in posts if p['uri'] not in seen_uris]
        for p in new_posts:
            seen_uris.add(p['uri'])

        if new_posts: # Salva os posts no CSV
            writer.writerows(new_posts) 
            total_collected += len(new_posts)

        if not cursor: 
            break

    print(f"[{query}] concluído. Total de posts coletados: {total_collected}")
    return total_collected

if __name__ == "__main__":
    # Configurações da busca
    MAX_QUERY_POSTS = 30000
    BATCH_SIZE = 100

    QUERIES = ["Vaccine"] 

    CSV_FILENAME = f"hackaton_vacina_{datetime.now().strftime('%Y%m%d')}.csv"

    # Login na conta bluesky
    load_dotenv()
    USERNAME = os.environ.get('BLUESKY_USERNAME')
    PASSWORD = os.environ.get('BLUESKY_PASSWORD')
    if not USERNAME or not PASSWORD:
        raise ValueError("Credenciais não encontradas no arquivo .env")


    client = Client()
    client.login(USERNAME, PASSWORD)

    seen_uris = set()

    total_saved = 0


    # Abre o CSV
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        writer = None
        

        for idx, query in enumerate(QUERIES):
            print(f"\n=== Iniciando coleta para: {query} ===")
            # Inicializa o writer com header na primeira vez
            if writer is None:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        'uri', 'text', 'author_handle', 'author_display_name',
                        'created_at', 'like_count', 'reply_count',
                        'repost_count', 'hashtags', 'url'
                    ],
                    quoting=csv.QUOTE_ALL,
                    quotechar='"'
                )
                writer.writeheader()

            collected = collect_for_query(query, BATCH_SIZE, MAX_QUERY_POSTS, seen_uris, writer)

    
    total_saved += collected
    print(f"\nBusca concluída. Total de posts salvos (únicos): {total_saved}")
    print(f"Arquivo: {CSV_FILENAME}")