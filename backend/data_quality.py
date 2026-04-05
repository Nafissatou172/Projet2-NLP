"""
Script de vérification, nettoyage et formatage du dataset finance personnelle.
Usage : python data_quality.py
"""

import pandas as pd
import re
import sys
import unicodedata
from collections import Counter
import csv

# ──────────────────────────────────────────────
#  1. CHARGEMENT
# ──────────────────────────────────────────────

INPUT_FILE = "dataset/dataset-finance-perso.csv"
OUTPUT_FILE = "dataset/dataset-finance-perso-clean.csv"
REPORT_FILE = "dataset/data_quality_report.txt"

print("=" * 70)
print("  VÉRIFICATION QUALITÉ DU DATASET FINANCE PERSONNELLE")
print("=" * 70)

try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"\n✅ Fichier chargé : {INPUT_FILE}")
except Exception as e:
    print(f"\n❌ Erreur de chargement : {e}")
    sys.exit(1)

report_lines = []


def log(msg: str):
    """Affiche et enregistre dans le rapport."""
    print(msg)
    report_lines.append(msg)


# ──────────────────────────────────────────────
#  2. APERÇU GÉNÉRAL
# ──────────────────────────────────────────────

log(f"\n{'─' * 70}")
log("  📊 APERÇU GÉNÉRAL")
log(f"{'─' * 70}")
log(f"  Nombre de lignes       : {len(df)}")
log(f"  Nombre de colonnes     : {len(df.columns)}")
log(f"  Colonnes détectées     : {list(df.columns)}")
log(f"  Types de données       :")
for col in df.columns:
    log(f"    - {col:25s} → {df[col].dtype}")


# ──────────────────────────────────────────────
#  3. VÉRIFICATION DES VALEURS MANQUANTES
# ──────────────────────────────────────────────

log(f"\n{'─' * 70}")
log("  🔍 VALEURS MANQUANTES")
log(f"{'─' * 70}")

missing = df.isnull().sum()
total_missing = missing.sum()

for col in df.columns:
    count = missing[col]
    pct = (count / len(df)) * 100
    status = "✅" if count == 0 else "⚠️"
    log(f"  {status} {col:25s} : {count:4d} manquantes ({pct:.1f}%)")

if total_missing > 0:
    log(f"\n  ⚠️  Total valeurs manquantes : {total_missing}")
    # Afficher les lignes avec des valeurs manquantes
    rows_with_missing = df[df.isnull().any(axis=1)]
    log(f"  Lignes concernées (indices) : {list(rows_with_missing.index)}")
else:
    log(f"\n  ✅ Aucune valeur manquante !")


# ──────────────────────────────────────────────
#  4. VÉRIFICATION DES DOUBLONS
# ──────────────────────────────────────────────

log(f"\n{'─' * 70}")
log("  🔄 DOUBLONS")
log(f"{'─' * 70}")

# Doublons exacts
exact_dupes = df.duplicated().sum()
log(f"  Doublons exacts (toutes colonnes) : {exact_dupes}")

# Doublons sur la question uniquement
if "question" in df.columns:
    question_dupes = df.duplicated(subset=["question"]).sum()
    log(f"  Questions en double              : {question_dupes}")
    if question_dupes > 0:
        duped_questions = df[df.duplicated(subset=["question"], keep=False)]
        log(f"  Questions dupliquées :")
        for q in duped_questions["question"].unique():
            count = len(duped_questions[duped_questions["question"] == q])
            log(f"    - ({count}x) {q[:80]}...")


# ──────────────────────────────────────────────
#  5. VÉRIFICATION DU CONTENU TEXTE
# ──────────────────────────────────────────────

log(f"\n{'─' * 70}")
log("  📝 QUALITÉ DU TEXTE")
log(f"{'─' * 70}")

issues_found = 0

for col in ["question", "reponse"]:
    if col not in df.columns:
        continue

    log(f"\n  ── Colonne : {col} ──")

    # Espaces en début/fin
    leading_trailing = df[col].dropna().apply(lambda x: x != x.strip()).sum()
    if leading_trailing > 0:
        log(f"  ⚠️  Espaces en début/fin : {leading_trailing} lignes")
        issues_found += leading_trailing
    else:
        log(f"  ✅ Pas d'espaces parasites en début/fin")

    # Doubles espaces
    double_spaces = df[col].dropna().apply(lambda x: "  " in x).sum()
    if double_spaces > 0:
        log(f"  ⚠️  Doubles espaces : {double_spaces} lignes")
        issues_found += double_spaces
    else:
        log(f"  ✅ Pas de doubles espaces")

    # Lignes vides ou trop courtes
    too_short = df[col].dropna().apply(lambda x: len(x.strip()) < 10).sum()
    if too_short > 0:
        log(f"  ⚠️  Textes trop courts (< 10 car.) : {too_short} lignes")
        short_texts = df[df[col].apply(lambda x: len(str(x).strip()) < 10 if pd.notna(x) else False)]
        for idx, row in short_texts.iterrows():
            log(f"       Ligne {idx}: \"{row[col]}\"")
        issues_found += too_short
    else:
        log(f"  ✅ Toutes les entrées ont une longueur suffisante")

    # Caractères spéciaux suspects
    special_chars = df[col].dropna().apply(
        lambda x: bool(re.search(r'[<>{}|\\~`]', x))
    ).sum()
    if special_chars > 0:
        log(f"  ⚠️  Caractères spéciaux suspects : {special_chars} lignes")
        issues_found += special_chars
    else:
        log(f"  ✅ Pas de caractères spéciaux suspects")

    # Stats de longueur
    lengths = df[col].dropna().apply(len)
    log(f"  📏 Longueur : min={lengths.min()}, max={lengths.max()}, "
        f"moy={lengths.mean():.0f}, méd={lengths.median():.0f}")


# ──────────────────────────────────────────────
#  6. VÉRIFICATION DES CATÉGORIES
# ──────────────────────────────────────────────

log(f"\n{'─' * 70}")
log("  🏷️  CATÉGORIES & NIVEAUX")
log(f"{'─' * 70}")

if "categorie" in df.columns:
    categories = df["categorie"].dropna().value_counts()
    log(f"\n  Catégories ({len(categories)} uniques) :")
    for cat, count in categories.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        log(f"    {cat:35s} : {count:4d} ({pct:5.1f}%) {bar}")

    # Vérifier la cohérence des noms de catégories
    cat_names = df["categorie"].dropna().unique()
    inconsistent = []
    for i, c1 in enumerate(cat_names):
        for c2 in cat_names[i + 1:]:
            if c1.lower().replace(" ", "") == c2.lower().replace(" ", "") and c1 != c2:
                inconsistent.append((c1, c2))
    if inconsistent:
        log(f"\n  ⚠️  Catégories potentiellement incohérentes :")
        for c1, c2 in inconsistent:
            log(f"       \"{c1}\" vs \"{c2}\"")

if "niveau_difficulte" in df.columns:
    niveaux = df["niveau_difficulte"].dropna().value_counts()
    log(f"\n  Niveaux de difficulté ({len(niveaux)} uniques) :")
    for niv, count in niveaux.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        log(f"    {str(niv):15s} : {count:4d} ({pct:5.1f}%) {bar}")

    # Vérifier les valeurs attendues
    expected_niveaux = {"Facile", "Moyen", "Difficile"}
    actual_niveaux = set(df["niveau_difficulte"].dropna().unique())
    unexpected = actual_niveaux - expected_niveaux
    if unexpected:
        log(f"\n  ⚠️  Niveaux inattendus trouvés : {unexpected}")


# ──────────────────────────────────────────────
#  7. VÉRIFICATION DE L'ENCODAGE / ACCENTS
# ──────────────────────────────────────────────

log(f"\n{'─' * 70}")
log("  🔤 ENCODAGE & ACCENTS")
log(f"{'─' * 70}")

accent_issues = 0
for col in ["question", "reponse"]:
    if col not in df.columns:
        continue
    for idx, val in df[col].dropna().items():
        # Chercher des séquences typiques d'encodage cassé (mojibake)
        if re.search(r'[Ã©Ã¨Ã Ã§Ã´Ã®Ã¹Ã»Ã¢]', str(val)):
            if accent_issues < 5:  # Montrer les 5 premiers
                log(f"  ⚠️  Ligne {idx} ({col}): encodage suspect → \"{val[:80]}...\"")
            accent_issues += 1

        # Chercher des accents manquants sur des mots français courants
        if re.search(r'\b(a ete|ete|echeance|interet|prelevement|epargne|credit)\b',
                      str(val).lower()):
            pass  # Note : certains mots sans accent peuvent être intentionnels

if accent_issues > 0:
    log(f"  ⚠️  Total problèmes d'encodage : {accent_issues}")
else:
    log(f"  ✅ Encodage correct (pas de mojibake détecté)")


# ══════════════════════════════════════════════
#  8. NETTOYAGE AUTOMATIQUE
# ══════════════════════════════════════════════

log(f"\n{'═' * 70}")
log("  🧹 NETTOYAGE AUTOMATIQUE")
log(f"{'═' * 70}")

df_clean = df.copy()
changes = 0

# 8a. Supprimer les espaces en début/fin
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        before = df_clean[col].copy()
        df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        changed = (before != df_clean[col]).sum()
        if changed > 0:
            log(f"  ✂️  Espaces trimés dans '{col}' : {changed} lignes")
            changes += changed

# 8b. Supprimer les doubles espaces
for col in ["question", "reponse"]:
    if col not in df_clean.columns:
        continue
    before = df_clean[col].copy()
    df_clean[col] = df_clean[col].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x
    )
    changed = (before != df_clean[col]).sum()
    if changed > 0:
        log(f"  ✂️  Espaces multiples nettoyés dans '{col}' : {changed} lignes")
        changes += changed

# 8c. Normaliser les guillemets
for col in ["question", "reponse"]:
    if col not in df_clean.columns:
        continue
    before = df_clean[col].copy()
    df_clean[col] = df_clean[col].apply(
        lambda x: x.replace("''", "'").replace('""', '"').replace("«", '"').replace("»", '"')
        if isinstance(x, str) else x
    )
    changed = (before != df_clean[col]).sum()
    if changed > 0:
        log(f"  ✂️  Guillemets normalisés dans '{col}' : {changed} lignes")
        changes += changed

# 8d. Supprimer les doublons exacts
before_len = len(df_clean)
df_clean = df_clean.drop_duplicates()
dupes_removed = before_len - len(df_clean)
if dupes_removed > 0:
    log(f"  🗑️  Doublons exacts supprimés : {dupes_removed}")
    changes += dupes_removed

# 8e. Supprimer les doublons sur la question (garder le premier)
if "question" in df_clean.columns:
    before_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["question"], keep="first")
    q_dupes_removed = before_len - len(df_clean)
    if q_dupes_removed > 0:
        log(f"  🗑️  Questions en double supprimées : {q_dupes_removed}")
        changes += q_dupes_removed

# 8f. Normaliser les catégories (capitalisation cohérente)
if "categorie" in df_clean.columns:
    before = df_clean["categorie"].copy()
    df_clean["categorie"] = df_clean["categorie"].apply(
        lambda x: x.strip().title() if isinstance(x, str) else x
    )
    changed = (before != df_clean["categorie"]).sum()
    if changed > 0:
        log(f"  🏷️  Catégories normalisées (Title Case) : {changed} lignes")
        changes += changed

# 8g. Normaliser les niveaux de difficulté
if "niveau_difficulte" in df_clean.columns:
    before = df_clean["niveau_difficulte"].copy()
    df_clean["niveau_difficulte"] = df_clean["niveau_difficulte"].apply(
        lambda x: x.strip().capitalize() if isinstance(x, str) else x
    )
    changed = (before != df_clean["niveau_difficulte"]).sum()
    if changed > 0:
        log(f"  🏷️  Niveaux normalisés (Capitalize) : {changed} lignes")
        changes += changed

# 8h. S'assurer que les questions finissent par '?'
if "question" in df_clean.columns:
    before = df_clean["question"].copy()
    df_clean["question"] = df_clean["question"].apply(
        lambda x: x.rstrip() + " ?" if isinstance(x, str) and not x.rstrip().endswith("?") else x
    )
    changed = (before != df_clean["question"]).sum()
    if changed > 0:
        log(f"  ❓ Questions sans '?' corrigées : {changed} lignes")
        changes += changed

# 8i. S'assurer que les réponses finissent par un point
if "reponse" in df_clean.columns:
    before = df_clean["reponse"].copy()
    df_clean["reponse"] = df_clean["reponse"].apply(
        lambda x: x.rstrip() + "."
        if isinstance(x, str) and not x.rstrip().endswith((".", "!", "?", ")", "]"))
        else x
    )
    changed = (before != df_clean["reponse"]).sum()
    if changed > 0:
        log(f"  📌 Réponses sans ponctuation finale corrigées : {changed} lignes")
        changes += changed

# 8j. Supprimer les lignes avec des valeurs manquantes critiques
if any(col in df_clean.columns for col in ["question", "reponse"]):
    before_len = len(df_clean)
    df_clean = df_clean.dropna(subset=["question", "reponse"])
    na_removed = before_len - len(df_clean)
    if na_removed > 0:
        log(f"  🗑️  Lignes sans question/réponse supprimées : {na_removed}")
        changes += na_removed

# Réindexer
df_clean = df_clean.reset_index(drop=True)


# ──────────────────────────────────────────────
#  9. RÉSUMÉ FINAL
# ──────────────────────────────────────────────

log(f"\n{'═' * 70}")
log("  📋 RÉSUMÉ FINAL")
log(f"{'═' * 70}")
log(f"  Lignes avant nettoyage  : {len(df)}")
log(f"  Lignes après nettoyage  : {len(df_clean)}")
log(f"  Lignes supprimées       : {len(df) - len(df_clean)}")
log(f"  Corrections appliquées  : {changes}")

if "categorie" in df_clean.columns:
    log(f"  Catégories finales      : {df_clean['categorie'].nunique()}")
if "niveau_difficulte" in df_clean.columns:
    log(f"  Niveaux finaux          : {list(df_clean['niveau_difficulte'].unique())}")

log(f"\n  📁 Fichier nettoyé sauvegardé : {OUTPUT_FILE}")


# ──────────────────────────────────────────────
#  10. SAUVEGARDE
# ──────────────────────────────────────────────

df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
print(f"\n✅ Dataset nettoyé sauvegardé dans : {OUTPUT_FILE}")

# Sauvegarder le rapport
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"✅ Rapport de qualité sauvegardé dans : {REPORT_FILE}")

print(f"\n{'═' * 70}")
print("  ✅ TERMINÉ")
print(f"{'═' * 70}")
