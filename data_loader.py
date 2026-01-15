#importing dependencies(langchain document,sitemap loader ...)
from langchain_community.document_loaders import SitemapLoader
import config
import kagglehub
import pandas as pd
import os
import json
import deeplake

def load_and_upload_to_activeloop(
    dataset_name="moaaztameer/medqa-usmle",
    deeplake_path="hub://noamaneoel/medqa-dataset"
):
    """
    Charge MedQA et l'upload directement sur ActiveLoop
    """
    
    # ========== ÉTAPE 1 : Charger les données ==========
    print("📥 Téléchargement MedQA...")
    path = kagglehub.dataset_download(dataset_name)
    
    questions = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(('.json', '.jsonl')):
                print(f"  📄 {file}")
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))
    
    print(f"✓ {len(questions)} questions chargées\n")
    
    # ========== ÉTAPE 2 : Créer le dataset ActiveLoop ==========
    print("🔮 Création du dataset ActiveLoop...")
    
    ds = deeplake.empty(deeplake_path, overwrite=True,token=config.token)
    
    # Définir le schéma

    ds.create_tensor('text', htype='text')
    ds.create_tensor('question', htype='text')
    ds.create_tensor('answer', htype='text')
    ds.create_tensor('id', htype='text')
    
    print("✓ Schéma créé")
    
    # ========== ÉTAPE 3 : Upload les données ==========
    print(f"\n📤 Upload de {len(questions)} questions...")
    
    with ds:
        for i, item in enumerate(questions):
            q = str(item.get('question', '')).strip()
            a = str(item.get('answer', '')).strip()
            
            if q and a and q != 'nan' and a != 'nan':
                text = f"Question: {q}\nAnswer: {a}"
                
                ds.text.append(text)
                ds.question.append(q)
                ds.answer.append(a)
                ds.id.append(str(item.get('id', f'q_{i}')))
            
            # Afficher progression
            if (i + 1) % 100 == 0:
                print(f"  ✓ {i + 1}/{len(questions)} uploadées")
    
    print(f"\n✅ Dataset ActiveLoop créé!")
    print(f"   Path: {deeplake_path}")
    print(f"   Taille: {len(ds)} entrées")
    
    return ds

# ========== UTILISATION ==========


ds = load_and_upload_to_activeloop()

# Vérifier
print("\n📊 Vérification:")
print(f"  Première question: {ds.question[0].numpy()[:100]}...")


#Adding authoritative sources via sitemap
sitemaps=["https://www.jmir.org/sitemap.xml","https://medlineplus.gov/sitemap.xml","https://revues.imist.ma/index.php/ReMaDiP/sitemap/"]
mayo_urls=SitemapLoader(web_path=[url for url in sitemaps],filter_urls="insert keyword variable based on user input here")[:200]

#storing pages from sitemap into a variable
mayo_docs=mayo_urls.load()
print(f"Loaded {len(mayo_docs)} documents from sitemaps")
for doc in mayo_docs[:2]:
    print(doc.page_content[:500])  # Print the first 500 characters of the first 2 documents