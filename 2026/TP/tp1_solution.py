import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Configuration
sns.set_theme(style="whitegrid", palette="muted")


def partie1_penguins_data_prep():
    """
    Partie 1 (1.1 à 1.5) : Chargement, Inspection, Manipulation, Nettoyage de base
    """
    print("--- PARTIE 1 : PENGUINS - PREPARATION ---")

    # 1.2 Chargement
    print("Chargement des données...")
    penguins_raw = fetch_openml(data_id=42585, as_frame=True, parser='auto')
    df = penguins_raw.frame

    # 1.3 Inspection
    print(f"Dimensions initiales : {df.shape}")
    print("\nTypes des variables :")
    print(df.dtypes)

    # 1.4 Manipulation (Filtre)
    gros_manchots = df[(df['island'] == 'Torgersen')
                       & (df['body_mass_g'] > 4000)]
    print(f"\nNombre de gros manchots sur Torgersen : {len(gros_manchots)}")

    # 1.5 Valeurs manquantes
    print(f"\nNaN avant nettoyage (Total) : {df.isna().sum().sum()}")

    mediane_masse = df['body_mass_g'].median()
    df['body_mass_g'] = df['body_mass_g'].fillna(mediane_masse)

    # On drop le reste
    df_clean = df.dropna()
    print(f"Dimensions après nettoyage : {df_clean.shape}")

    return df_clean


def partie1_penguins_advanced_ops(df):
    """
    Partie 1 (1.6 à 1.8) : Feature Engineering, Outliers, Standardisation
    """
    print("\n--- PARTIE 1 : PENGUINS - OPERATIONS AVANCEES ---")
    df_adv = df.copy()

    # 1.6 Feature Engineering (Apply)
    def categoriser_poids(masse):
        if masse < 3500:
            return 'Léger'
        elif masse <= 4500:
            return 'Moyen'
        else:
            return 'Lourd'

    df_adv['weight_category'] = df_adv['body_mass_g'].apply(
        lambda x: categoriser_poids(x))
    print("Distribution des catégories de poids :")
    print(df_adv['weight_category'].value_counts())

    # 1.7 Outliers (IQR)
    Q1 = df_adv['culmen_length_mm'].quantile(0.25)
    Q3 = df_adv['culmen_length_mm'].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    outliers = df_adv[(df_adv['culmen_length_mm'] < borne_inf)
                      | (df_adv['culmen_length_mm'] > borne_sup)]
    print(
        f"\nDétection Outliers (Culmen Length) : {len(outliers)} individus hors de [{borne_inf:.2f}, {borne_sup:.2f}]")

    # 1.8 Standardisation (Z-score)
    mean_mass = df_adv['body_mass_g'].mean()
    std_mass = df_adv['body_mass_g'].std()
    df_adv['body_mass_g_scaled'] = (
        df_adv['body_mass_g'] - mean_mass) / std_mass

    print(
        f"Moyenne de body_mass_g_scaled : {df_adv['body_mass_g_scaled'].mean():.4f} (doit être proche de 0)")

    return df_adv


def partie1_penguins_visu_encodage(df):
    """
    Partie 1 (1.9 à 1.10) : Visualisation avancée et Encodage
    """
    print("\n--- PARTIE 1 : PENGUINS - VISUALISATION AVANCEE & ENCODAGE ---")

    # 1. Visualisation : Jointplot (Scatter + Histogrammes marginaux)
    # Montre la relation entre la longueur et la profondeur du bec, avec les distributions sur les côtés
    g = sns.jointplot(data=df, x='culmen_length_mm',
                      y='culmen_depth_mm', hue='species', kind='scatter', height=7)
    g.fig.suptitle(
        "[Correction] Jointplot : Relation & Distribution du Bec", y=1.02)
    plt.show()

    # 2. Visualisation : Violin Plot "Split" (Comparaison Mâle/Femelle)
    # Permet de voir la distribution de la masse par espèce, séparée par sexe sur le même axe
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='species', y='body_mass_g', hue='sex',
                   split=True, inner="quartile", palette="Pastel1")
    plt.title(
        "[Correction] Violinplot : Distribution de la masse (Comparaison Mâle/Femelle)")
    plt.show()

    # 3. Replot: facettes
    # Il permet de diviser le graphique en plusieurs sous-graphiques selon une variable catégorielle (ici l'île) grâce à l'argument `col`.
    sns.relplot(data=df, x='culmen_length_mm',
                y='culmen_depth_mm', hue='species', col='island')
    plt.show()

    # 4. Visualisation : Pairplot (La vue d'ensemble ultime)
    print("Génération du Pairplot (peut prendre quelques secondes)...")
    sns.pairplot(df, hue='species', vars=['culmen_length_mm', 'culmen_depth_mm',
                 'flipper_length_mm', 'body_mass_g'], markers=["o", "s", "D"], corner=True)
    plt.suptitle(
        "[Correction] Pairplot : Vue d'ensemble des variables biométriques", y=1.02)
    plt.show()

    # 4. Encodage final
    df_final = df.copy()
    df_final['sex_encoded'] = df_final['sex'].map({'FEMALE': 0, 'MALE': 1})
    # Dummies Island
    df_ml_ready = pd.get_dummies(df_final, columns=['island'], drop_first=True)
    print(df_ml_ready.head())

    print("Colonnes finales pour le ML :", df_ml_ready.columns.tolist())
    return df_ml_ready


def partie2_titanic_solution():
    """
    Partie 2 : Titanic
    """
    print("\n" + "="*40)
    print("--- PARTIE 2 : TITANIC (SOLUTION ETUDIANT) ---")
    print("="*40)

    # --- ETAPES PRECEDENTES (Chargement / Nettoyage) ---
    titanic_raw = fetch_openml(data_id=40945, as_frame=True, parser='auto')
    df_titanic = titanic_raw.frame

    # Conversion & Nettoyage
    df_titanic['survived'] = pd.to_numeric(df_titanic['survived'])
    df_titanic = df_titanic.drop(
        columns=['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'])

    # Imputation Age
    mediane_age = df_titanic['age'].median()
    df_titanic['age'] = df_titanic['age'].fillna(mediane_age)

    df_titanic_clean = df_titanic.dropna()

    # Important : On force le type 'int' pour que le sexe apparaisse dans la matrice de corrélation
    df_titanic_clean['sex'] = df_titanic_clean['sex'].map(
        {'male': 0, 'female': 1}).astype(int)

    df_titanic_ml = pd.get_dummies(df_titanic_clean, columns=[
        'embarked'], drop_first=True)

    # --- VISUALISATION AMELIOREE ---
    print("\n[Visualisation] Génération des graphes Titanic...")

    # 1. Histogramme des âges (Explication du pic)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_titanic_clean, x='age', hue='survived',
                 multiple="stack", kde=True, palette="viridis")
    plt.title(
        f"[Correction] Distribution des âges (Le pic à {mediane_age} ans est dû à l'imputation par la médiane)")
    plt.xlabel("Âge (les valeurs manquantes ont été remplacées par la médiane)")
    plt.show()

    # 2. Pivot Table
    pivot = pd.pivot_table(df_titanic_clean, values='survived',
                           index='pclass', columns='sex', aggfunc='mean')
    print("\nTableau croisé (Taux de survie par Classe et Sexe) :")
    print(pivot)

    # 3. Matrice de Corrélation
    # On crée une copie temporaire juste pour avoir des noms de colonnes lisibles sur le graphique
    df_viz = df_titanic_ml.copy()

    # Dictionnaire de renommage pour la clarté pédagogique
    rename_dict = {
        'survived': 'Survivant',
        'pclass': 'Classe (1=Riche)',
        'sex': 'Sexe (1=Femme)',
        'age': 'Age',
        'sibsp': 'Famille (Frère/Epoux)',
        'parch': 'Famille (Parent/Enfant)',
        'fare': 'Tarif Billet'
    }
    df_viz = df_viz.rename(columns=rename_dict)

    # Sélection des colonnes numériques
    vars_num = df_viz.select_dtypes(include=[np.number]).columns

    plt.figure(figsize=(10, 8))
    # Utilisation de cmap='RdBu' (Red-Blue) centré sur 0 pour bien voir les corrélations positives (Bleu) et négatives (Rouge)
    sns.heatmap(df_viz[vars_num].corr(), annot=True,
                cmap='RdBu', center=0, fmt=".2f", linewidths=0.5)
    plt.title("[Correction] Matrice de corrélation (Variables renommées)")
    plt.show()

    print("\nAnalyse Corrélation :")
    print("- 'Sexe (1=Femme)' est fortement corrélé positivement à 'Survivant' (0.53) -> Les femmes survivent plus.")
    print("- 'Classe' est corrélée négativement (-0.31) -> Plus la classe augmente (3ème classe), moins on survit.")


if __name__ == "__main__":
    # Exécution séquentielle
    df_penguins = partie1_penguins_data_prep()
    df_penguins_adv = partie1_penguins_advanced_ops(df_penguins)
    df_penguins_final = partie1_penguins_visu_encodage(df_penguins_adv)

    partie2_titanic_solution()
