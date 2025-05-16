import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# Chargement des données
print("📥 Chargement des données depuis Data.json ...")
df = pd.read_json("Data.json", lines=True)

print(f"🔢 Nombre de lignes AVANT nettoyage : {len(df)}")

# Supprimer les lignes avec valeurs manquantes
df = df.dropna(subset=["reviewText", "overall"])

print(f"🧹 Nombre de lignes APRÈS suppression valeurs vides : {len(df)}")

# Fonction de nettoyage texte
def clean_text(text):
    text = text.lower()  # passage en minuscules
    # suppression ponctuation sauf chiffres (on garde les chiffres pour l'instant)
    text = re.sub(r"[^\w\s\d]", " ", text)
    # remplacement des espaces multiples par un seul espace, suppression des espaces en début/fin
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Nettoyage des colonnes texte
df["reviewText"] = df["reviewText"].astype(str).apply(clean_text)
df["summary"] = df["summary"].astype(str).apply(clean_text)
df["reviewerName"] = df["reviewerName"].astype(str).apply(clean_text)

# Création de la colonne "label" pour la classification
def to_label(score):
    if score < 3:
        return 0  # négatif
    elif score == 3:
        return 1  # neutre
    else:
        return 2  # positif

df["label"] = df["overall"].apply(to_label)

# Affichage de la répartition des classes
class_counts = df["label"].value_counts().sort_index()
print("\n📊 Répartition des classes après nettoyage :")
print(f" - Négatif (0) : {class_counts.get(0, 0)}")
print(f" - Neutre  (1) : {class_counts.get(1, 0)}")
print(f" - Positif (2) : {class_counts.get(2, 0)}")

# === Séparation Train (80%) - Temp (20%) ===
# On stratifie pour garder la même proportion des classes dans chaque sous-ensemble.
X_train, X_temp, y_train, y_temp = train_test_split(
    df, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# === Séparation Temp (20%) en Validation (10%) et Test (10%) ===
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n🔖 Nombre lignes train : {len(X_train)}")
print(f"🔖 Nombre lignes validation : {len(X_val)}")
print(f"🔖 Nombre lignes test : {len(X_test)}")

# Sauvegarde des fichiers
os.makedirs("code", exist_ok=True)

# Train et validation gardent la colonne label car ils sont supervisés
X_train.to_json("code/train_data.json", orient="records", lines=True, force_ascii=False)
X_val.to_json("code/validation_data.json", orient="records", lines=True, force_ascii=False)

# Pour le test final (flux), on supprime la colonne label pour simuler un test non supervisé
X_test_nolabel = X_test.drop(columns=["label"])
X_test_nolabel.to_json("code/test_data_nolabel.json", orient="records", lines=True, force_ascii=False)

print("\n✅ Données sauvegardées :")
print(" - Entraînement : code/train_data.json")
print(" - Validation  : code/validation_data.json")
print(" - Test final (sans label) : code/test_data_nolabel.json")
