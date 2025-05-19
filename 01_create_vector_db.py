from scrape_insee import scrape_urls
from chunking import create_chunks
from embed_and_store import embed_and_store_chunks

urls = [
    'https://www.insee.fr/fr/statistiques/8569094',
    'https://www.insee.fr/fr/statistiques/8570040',
    'https://www.insee.fr/fr/statistiques/8569615'
]

if __name__ == "__main__":
    documents = []

    # documents = scrape_urls(urls, do_cache=True)
    documents = scrape_urls(urls, use_cache=True)

    # On chunk nos documents
    all_chunks = create_chunks(documents)

    # On fait appel au modèle d'embeddings et on crée notre base vectorielle
    embed_and_store_chunks(all_chunks)
