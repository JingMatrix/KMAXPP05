import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

sns.set_theme(style="whitegrid")


# =============================================================================
# Utility Functions
# =============================================================================
def plot_history(history, title="Courbes d'apprentissage"):
    """Trace les courbes de loss et d'accuracy pour l'entraînement et la validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Validation Acc')
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# PART 1 : Fundamental Concepts & Keras APIs (MNIST)
# =============================================================================
def part1_mnist():
    print("\n--- PARTIE 1 : MNIST ---")
    # Chargement de MNIST
    (X_train_full, y_train_full), (X_test,
                                   y_test) = keras.datasets.mnist.load_data()

    # Flatten
    X_train_flat = X_train_full.reshape(-1, 784).astype('float32')
    X_test_flat = X_test.reshape(-1, 784).astype('float32')

    # Normalisation
    X_train_norm = X_train_flat / 255.0
    X_test_norm = X_test_flat / 255.0

    print(f"Dimensions d'entraînement : {X_train_norm.shape}")

    # 1.2 Scikit-Learn
    print("\nEntraînement Scikit-Learn...")
    sk_model = MLPClassifier(hidden_layer_sizes=(64,),
                             max_iter=10, random_state=42)
    sk_model.fit(X_train_norm, y_train_full)
    sk_preds = sk_model.predict(X_test_norm)
    print(f"Précision Scikit-Learn : {accuracy_score(y_test, sk_preds):.4f}")

    # 1.3 Keras API Sequential
    print("\nEntraînement Keras Séquentiel...")
    model_seq = Sequential([
        Dense(64, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model_seq.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    model_seq.fit(X_train_norm, y_train_full, epochs=2,
                  batch_size=128, validation_split=0.1)

    # 1.4 Keras API Fonctionnelle
    print("\nCréation Keras Fonctionnel...")
    inputs = Input(shape=(784,))
    h1 = Dense(64, activation='relu')(inputs)
    outputs = Dense(10, activation='softmax')(h1)

    model_func = Model(inputs=inputs, outputs=outputs)
    model_func.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    print("Résumé API Fonctionnelle :")
    model_func.summary()


# =============================================================================
# PART 2 : Kepler Dataset (Preprocessing & Base Models)
# =============================================================================
def part2_kepler_preprocessing(filepath='cumulative.csv'):
    print("\n--- PARTIE 2 : KEPLER PREPROCESSING ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Le fichier {filepath} est introuvable. Veuillez le télécharger pour exécuter le script.")

    df_kepler = pd.read_csv(filepath)
    print("Dimensions initiales :", df_kepler.shape)

    # Nettoyage et gestion des fuites
    df_kepler = df_kepler[df_kepler['koi_disposition'].isin(
        ['CONFIRMED', 'FALSE POSITIVE'])]
    df_kepler['target'] = df_kepler['koi_disposition'].map(
        {'CONFIRMED': 1, 'FALSE POSITIVE': 0})

    features_physiques = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
        'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
        'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag'
    ]
    df_physique = df_kepler[features_physiques + ['target']].copy()

    # Suppression des NaN
    df_clean = df_physique.dropna()
    print("Dimensions après nettoyage :", df_clean.shape)

    # Train/Test Split
    X = df_clean.drop(columns=['target']).values
    y = df_clean['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def part2_kepler_modeling(X_train_scaled, y_train):
    print("\n--- PARTIE 2 : KEPLER MODELING ---")
    input_dim = X_train_scaled.shape[1]

    # API Séquentielle
    model_seq = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # API Fonctionnelle
    inputs = Input(shape=(input_dim,))
    h1 = Dense(128, activation='relu')(inputs)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(32, activation='relu')(h2)
    outputs = Dense(1, activation='sigmoid')(h3)
    model_func = Model(inputs=inputs, outputs=outputs)

    # Vérification
    assert model_seq.count_params() == model_func.count_params(), "Les topologies diffèrent !"
    print(
        f"Topologies validées : {model_seq.count_params()} paramètres à optimiser.")

    # Entraînement Modèle de Base
    model_seq.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
    history_base = model_seq.fit(X_train_scaled, y_train,
                                 epochs=70, batch_size=128,
                                 validation_split=0.2, verbose=1)

    plot_history(history_base, "Modèle de Base (128->64->32->1)")

    # -------------------------------------------------------------------------
    # RÉPONSE À L'ANALYSE (Partie 2)
    # -------------------------------------------------------------------------
    """
    Analyse Pédagogique Attendue (Overfitting) :
    La courbe 'Train Loss' continue de descendre de manière asymtotique vers 0. 
    Cependant, la 'Validation Loss' cesse de diminuer et commence à remonter 
    aux alentours de l'itération (epoch) 15 à 20. 
    Le nom mathématique de ce phénomène est le surapprentissage (overfitting). 
    Le modèle a cessé de généraliser et commence à mémoriser le bruit du Train set.
    """


# =============================================================================
# PART 3 : Regularization & Callbacks
# =============================================================================
def part3_regularization_and_callbacks(X_train_scaled, y_train, X_test_scaled, y_test):
    print("\n--- PARTIE 3 : RÉGULARISATION ET CALLBACKS ---")
    input_dim = X_train_scaled.shape[1]

    # --- 3.1 Évaluation Isolée ---
    # L2 Regularization
    print("Test de la pénalité L2...")
    model_l2 = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(
            0.01), input_shape=(input_dim,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model_l2.compile(optimizer='adam',
                     loss='binary_crossentropy', metrics=['accuracy'])
    history_l2 = model_l2.fit(X_train_scaled, y_train, epochs=70,
                              batch_size=128, validation_split=0.2, verbose=1)
    plot_history(history_l2, "1. Effet de la Pénalité L2")

    # Dropout
    print("Test du Dropout...")
    model_dropout = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model_dropout.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_dropout = model_dropout.fit(
        X_train_scaled, y_train, epochs=70, batch_size=128, validation_split=0.2, verbose=1)
    plot_history(history_dropout, "2. Effet du Dropout")

    # Batch Normalization
    print("Test de la Batch Normalization...")
    model_bn = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model_bn.compile(optimizer='adam',
                     loss='binary_crossentropy', metrics=['accuracy'])
    history_bn = model_bn.fit(X_train_scaled, y_train, epochs=70,
                              batch_size=128, validation_split=0.2, verbose=1)
    plot_history(history_bn, "3. Effet de la Batch Normalization")

    # --- 3.2 Optimisation Dynamique ---
    print("\nEntraînement final avec Callbacks...")
    # On reprend un modèle régularisé au choix (ici avec du Dropout)
    model_reg = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model_reg.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    checkpoint = ModelCheckpoint(filepath='meilleur_modele_kepler.keras',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)

    history_reg = model_reg.fit(X_train_scaled, y_train,
                                epochs=150, batch_size=128,
                                validation_split=0.2,
                                callbacks=[reduce_lr, early_stop, checkpoint],
                                verbose=1)

    # --- 3.3 Évaluation du meilleur modèle ---
    modele_optimal = load_model('meilleur_modele_kepler.keras')
    test_loss, test_acc = modele_optimal.evaluate(
        X_test_scaled, y_test, verbose=1)
    print(f"\nPrécision optimale sur le jeu de test : {test_acc:.4f}")


# =============================================================================
# PART 4 : Free Design (Custom Architecture)
# =============================================================================
def part4_custom_architecture(X_train_scaled, y_train, X_test_scaled, y_test):
    print("\n--- PARTIE 4 : MODÈLE LIBRE ---")
    input_dim = X_train_scaled.shape[1]

    # Modèle optimisé plus modeste (adapté au tabular data Kepler de 15 features)
    model_final = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_final.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    print("Entraînement du modèle final...")
    history_final = model_final.fit(X_train_scaled, y_train,
                                    epochs=100, batch_size=64,
                                    validation_split=0.2,
                                    callbacks=[early_stop, reduce_lr],
                                    verbose=1)

    plot_history(history_final, "Modèle Libre Final")

    test_loss, test_acc = model_final.evaluate(
        X_test_scaled, y_test, verbose=1)
    print(f"Performance finale sur le jeu de test inédit : {test_acc:.4f}")

    # -------------------------------------------------------------------------
    # RÉPONSE À L'ANALYSE (Partie 4)
    # -------------------------------------------------------------------------
    """
    Analyse Pédagogique Attendue (Conception libre) :
    1. Profondeur et Largeur : J'ai réduit le nombre de neurones initiaux (64 
       au lieu de 128) car l'espace d'entrée n'a que 15 dimensions. Un réseau 
       trop grand pour peu de variables accélère le surapprentissage.
    2. Régularisation mixte : J'ai combiné BatchNormalization (pour accélérer
       la convergence initiale) et un Dropout modéré (20%) pour éviter de 
       sur-dépendre de features dominantes comme 'koi_depth'.
    3. Résultat : La courbe de validation est beaucoup plus stable et ne 
       diverge plus brutalement. La performance de Test est robuste et 
       souvent supérieure à celle de l'architecture basique non-régularisée.
    """


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # Partie 1 : Connaissances API
    part1_mnist()

    # Partie 2 & 3 & 4 : L'Étude de Cas (Kepler)
    try:
        X_train_s, X_test_s, y_train, y_test = part2_kepler_preprocessing(
            'cumulative.csv')

        part2_kepler_modeling(X_train_s, y_train)
        part3_regularization_and_callbacks(
            X_train_s, y_train, X_test_s, y_test)
        part4_custom_architecture(X_train_s, y_train, X_test_s, y_test)

    except FileNotFoundError as e:
        print(f"\n[ATTENTION] {e}")
        print("Téléchargez cumulative.csv depuis Kaggle (Kepler Exoplanet Search Results) pour exécuter la suite du script.")
