import csv
import os
from datetime import datetime, timezone
from collections import defaultdict
from atproto import Client
from dotenv import load_dotenv


# ========== CONFIGURAÇÕES ==========
load_dotenv()
CSV_FILENAME = "hackaton_vacinaPT_20250920.csv"
USERNAME = os.environ.get("BLUESKY_USERNAME")
PASSWORD = os.environ.get("BLUESKY_PASSWORD")

client = Client()
client.login(USERNAME, PASSWORD)

# ========== FUNÇÕES AUXILIARES ==========
def get_users_metadata(handles, batch_size=25):
    """Busca metadados de vários usuários em lotes"""
    metadata = {}
    for i in range(0, len(handles), batch_size):
        batch = handles[i:i+batch_size]
        try:
            response = client.app.bsky.actor.get_profiles({"actors": batch})
            for profile in response.profiles:
                metadata[profile.handle] = {
                    "followers": profile.followers_count,
                    "following": profile.follows_count,
                    "created_at": profile.created_at
                }
        except Exception as e:
            print(f"Erro ao buscar batch {batch}: {e}")
    return metadata


def evaluate_user_flags(user, posts, duplicate_texts):
    """Avalia as três condições e retorna flags booleanas"""
    flags = {
        "is_recent_account": False,
        "is_following_ratio_high": False,
        "has_duplicate_posts": False
    }

    meta = user["metadata"]

    # conta < 30 dias
    if meta["created_at"]:
        created = datetime.fromisoformat(meta["created_at"].replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).days
        if age_days < 30:
            flags["is_recent_account"] = True

    # seguindo/seguidores > 8
    if meta["followers"] > 0 and meta["following"] / meta["followers"] > 8:
        flags["is_following_ratio_high"] = True

    # posts duplicados
    for p in posts:
        if p["text"] in duplicate_texts:
            flags["has_duplicate_posts"] = True
            break

    return flags


# ========== PROCESSAMENTO ==========
# 1. Lê posts do CSV
user_posts = defaultdict(list)
all_texts = defaultdict(list)

with open(CSV_FILENAME, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_posts[row["author_handle"]].append(row)
        all_texts[row["text"]].append(row["author_handle"])

# 2. Detecta textos duplicados
duplicate_texts = {text for text, authors in all_texts.items() if len(authors) > 1}

# 3. Busca metadados em lote
all_handles = list(user_posts.keys())
users_metadata = get_users_metadata(all_handles)

results = []
for handle, posts in user_posts.items():
    meta = users_metadata.get(handle, {"followers": 0, "following": 0, "created_at": None})
    user_info = {"handle": handle, "num_posts": len(posts), "metadata": meta}
    flags = evaluate_user_flags(user_info, posts, duplicate_texts)
    results.append({
        "handle": handle,
        "num_posts": len(posts),
        "followers": meta["followers"],
        "following": meta["following"],
        "created_at": meta["created_at"],
        **flags
    })

# 4. Salva resultado
OUTPUT_FILENAME = "usuarios_suspeitos.csv"
with open(OUTPUT_FILENAME, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Análise concluída. Resultados em: {OUTPUT_FILENAME}")

with open("posts_suspeitos.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["handle", "text", "is_recent_account", "is_following_ratio_high", "has_duplicate_posts"]
    )
    writer.writeheader()
    for r in results:
        for p in user_posts[r["handle"]]:
            writer.writerow({
                "handle": r["handle"],
                "text": p["text"],
                "is_recent_account": r["is_recent_account"],
                "is_following_ratio_high": r["is_following_ratio_high"],
                "has_duplicate_posts": r["has_duplicate_posts"]
            })
