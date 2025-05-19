import json
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path

CACHE_PATH = Path("documents_cache.json")
urls = [
    'https://www.insee.fr/fr/statistiques/8569094',
    'https://www.insee.fr/fr/statistiques/8570040',
    'https://www.insee.fr/fr/statistiques/8569615'
]


def scrape_page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        content = page.content()
        browser.close()

    soup = BeautifulSoup(content, "html.parser").find("main")

    # Titres publication
    titre_principal = soup.find('h1', class_='titre-principal')
    titre_publication = titre_principal.find('span', class_='titre-titre').get_text(strip=True)
    sous_titre_publication = titre_principal.find('span', class_='sous-titre').get_text(strip=True)

    # Corps publication
    corps_publication_div = soup.find('div', class_='corps-publication')
    contenu_elements = corps_publication_div.select('h2, p')

    contenu = []
    for elem in contenu_elements:
        if elem.name == 'h2':
            section = elem.get_text(strip=True)
            contenu.append(f"\nSection : {section}")
        else:
            contenu.append(elem.get_text(strip=True))
    texte_final = '\n'.join(contenu)

    return {
        'titre': titre_publication,
        'sous_titre': sous_titre_publication,
        'contenu': texte_final
    }


def scrape_urls(urls, do_cache=False, use_cache=False):
    documents = []

    if use_cache:
        print("🕒 Récupération des publications depuis le cache...")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        print("🕒 Début du scraping des publications...")
        for url in tqdm(urls, desc="Scraping des pages Insee"):
            doc = scrape_page(url)
            documents.append(doc)

        if do_cache:
            print("\n📦 Sauvegarde des documents en cache...")
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)

        print("\n✅ Fin du scraping\n" + "=" * 80)

    return documents


if __name__ == "__main__":
    documents = []
    # documents = scrape_urls(urls, do_cache=True)
    documents = scrape_urls(urls, use_cache=True)

    print("\n📚 Résultat du scraping :\n" + "=" * 80)

    for idx, doc in enumerate(documents, start=1):
        print(f"\nPublication {idx}: {doc['titre']}")
        print(f"Sous-titre: {doc['sous_titre']}\n")
        print(f"{doc['contenu'][:1000]}...")  # Limite l'affichage aux 1000 premiers caractères
        print("\n" + "=" * 80)
