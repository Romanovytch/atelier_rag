from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=300,     # taille du chunk (en tokens)
    chunk_overlap=30,   # chevauchement entre chunks (en tokens)
    encoding_name="cl100k_base"  # tokenizer de GPT-3.5 / GPT-4
)


def create_chunks(documents):
    all_chunks = []

    for doc in documents:
        chunks = text_splitter.split_text(doc['contenu'])
        # On associe chaque chunk à ses métadonnées
        for chunk in chunks:
            chunk_with_meta = {
                'contenu': chunk,
                'titre': doc['titre'],
                'sous_titre': doc['sous_titre']
                # Tu pourrais même ajouter l'URL ici si tu le souhaites !
            }
            all_chunks.append(chunk_with_meta)

    print(f"Nombre total de chunks créés : {len(all_chunks)}\n")

    return all_chunks


if __name__ == "__main__":
    from scrape_insee import scrape_urls, urls

    documents = []
    documents = scrape_urls(urls, use_cache=True)

    all_chunks = create_chunks(documents)

    # Vérification simple du résultat
    for i, chunk in enumerate(all_chunks[:5], 1):
        print(f"Chunk {i} [Publication : {chunk['titre']}]\n{'='*60}\n{chunk['contenu']}\n{'='*60}")
