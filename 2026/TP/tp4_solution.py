# ==============================================================================
# TP4 : Modélisation Bayésienne, Intervalles de Confiance et Théorie de la Décision
# Fichier de conception - Enseignant
# ==============================================================================

from sklearn.metrics import roc_curve, auc
from scipy.stats import norm, poisson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from scipy.special import gammaln  # Pour calculer log(x!) sans overflow

# ==============================================================================
# PARTIE 1 : MODÉLISATION BAYÉSIENNE (Exoplanètes - Kepler)
# ==============================================================================


def load_and_prep_kepler(filepath="cumulative.csv"):
    """
    Charge et prépare les données Kepler en respectant les lois du TD :
    - x1 (Bernoulli) : Binarisation dynamique de la profondeur de l'éclipse.
    - x2 (Poisson) : Arrondi de la durée du transit (comptage d'heures).
    - x3 (Gaussienne) : Logarithme de la température.
    """
    print("--- Préparation des données Kepler ---")
    df = pd.read_csv(filepath)

    # 1. Filtrage strict : on exclut les CANDIDATE pour un problème binaire propre
    df = df[df['koi_disposition'] != 'CANDIDATE'].copy()
    df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)

    # =========================================================
    # x1 (Bernoulli) : Éclipse secondaire (via koi_depth)
    # =========================================================
    # On trouve le seuil optimal dynamiquement au lieu de le coder en dur
    depth = df['koi_depth'].fillna(
        df['koi_depth'].median()).values.reshape(-1, 1)

    tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    tree.fit(depth, df['target'])
    threshold_x1 = tree.tree_.threshold[0]

    df['x1_eclipse'] = (depth.flatten() > threshold_x1).astype(int)
    print(
        f"[*] x1_eclipse : Seuil optimal trouvé pour koi_depth = {threshold_x1:.2f} ppm")

    # =========================================================
    # x2 (Poisson) : Durée du transit (via koi_duration)
    # =========================================================
    # La durée en heures est arrondie à l'entier le plus proche pour
    # obtenir une variable de comptage (naturellement adaptée à Poisson)
    duree = df['koi_duration'].fillna(df['koi_duration'].median())
    df['x2_duree'] = np.round(duree).astype(int)

    # =========================================================
    # x3 (Gaussienne) : Température (via log(koi_teq))
    # =========================================================
    # Le passage au logarithme permet d'obtenir des distributions
    # beaucoup plus proches de l'hypothèse Gaussienne du TD
    teq = df['koi_teq'].fillna(df['koi_teq'].median())
    # On ajoute +1 pour éviter un éventuel log(0)
    df['x3_temp'] = np.log(teq + 1)

    # Nettoyage final
    df_clean = df[['x1_eclipse', 'x2_duree', 'x3_temp', 'target']].copy()
    print(df_clean.info())

    return df_clean


def explore_and_check_assumptions(df):
    """
    Affiche les corrélations conditionnelles (vérification Naïve Bayes)
    et trace les densités empiriques.
    """
    print("\n--- Vérification de l'hypothèse d'indépendance (Naïve Bayes) ---")
    print("Corrélation Spearman (Exoplanètes - Target 1) :")
    print(df[df['target'] == 1]
          [['x1_eclipse', 'x2_duree', 'x3_temp']].corr(method='spearman'))

    print("\nCorrélation Spearman (Naines/Anomalies - Target 0) :")
    print(df[df['target'] == 0]
          [['x1_eclipse', 'x2_duree', 'x3_temp']].corr(method='spearman'))

    print("\n-> Note aux étudiants : Si les corrélations sont proches de 0, l'hypothèse 'Naïve' est valide.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # x1 : Bernoulli
    sns.countplot(data=df, x='x1_eclipse', hue='target', ax=axes[0])
    axes[0].set_title("x1 : Éclipse Secondaire (Binarisée)")

    # x2 : Poisson (Durée)
    sns.histplot(data=df, x='x2_duree', hue='target', binwidth=1,
                 stat='density', common_norm=False, ax=axes[1])
    axes[1].set_xlim(0, 20)
    axes[1].set_title("x2 : Durée du transit (Heures)")

    # x3 : Gaussienne (Log Température)
    sns.kdeplot(data=df, x='x3_temp', hue='target',
                common_norm=False, ax=axes[2])
    axes[2].set_title("x3 : Log(Température) (KDE)")

    plt.tight_layout()
    plt.show()


def fit_mle_kepler(df):
    """
    Calcule les estimateurs du maximum de vraisemblance (MLE) pour chaque paramètre.
    """
    params = {}
    for c in [0, 1]:  # 0 = N, 1 = E
        sub = df[df['target'] == c]
        params[c] = {
            'p': sub['x1_eclipse'].mean(),  # MLE Bernoulli
            'lambda': sub['x2_duree'].mean(),  # MLE Poisson
            'mu': sub['x3_temp'].mean(),  # MLE Gaussienne (Moyenne)
            'sigma': sub['x3_temp'].std()  # MLE Gaussienne (Ecart-type)
        }
    return params


def compute_log_likelihoods(X, params_c):
    """
    Implémentation mathématique des log-vraisemblances.
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    p = np.clip(params_c['p'], 1e-9, 1-1e-9)  # Stabilité numérique
    lam = max(params_c['lambda'], 1e-9)
    mu, sigma = params_c['mu'], max(params_c['sigma'], 1e-9)

    # 1. Bernoulli: log(p^x * (1-p)^(1-x)) = x*log(p) + (1-x)*log(1-p)
    ll_x1 = x1 * np.log(p) + (1 - x1) * np.log(1 - p)

    # 2. Poisson: log(lambda^x * e^-lambda / x!) = x*log(lambda) - lambda - log(x!)
    # On utilise gammaln(x+1) qui équivaut à log(x!)
    ll_x2 = x2 * np.log(lam) - lam - gammaln(x2 + 1)

    # 3. Gaussienne: log(1/(sigma*sqrt(2pi)) * exp(-0.5 * ((x-mu)/sigma)^2))
    ll_x3 = -np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * ((x3 - mu) / sigma)**2

    return ll_x1 + ll_x2 + ll_x3


def predict_naive_bayes(X, params, prior_E):
    """
    Algorithme Naïve Bayes complet utilisant la fonction softmax.
    """
    prior_N = 1.0 - prior_E

    # Log-vraisemblances totales + Log-a priori
    log_prob_N = compute_log_likelihoods(X, params[0]) + np.log(prior_N)
    log_prob_E = compute_log_likelihoods(X, params[1]) + np.log(prior_E)

    # Application du Softmax : P(E|X) = exp(L_E) / (exp(L_E) + exp(L_N))
    # Astuce numérique (log-sum-exp) pour éviter l'overflow
    max_log = np.maximum(log_prob_E, log_prob_N)
    exp_E = np.exp(log_prob_E - max_log)
    exp_N = np.exp(log_prob_N - max_log)

    prob_E = exp_E / (exp_E + exp_N)
    return prob_E

# ==============================================================================
# PARTIE 2 : INTERVALLES DE CONFIANCE (Bootstrap) ET RÉGION DE PRÉDICTION
# ==============================================================================


def load_and_prep_credit():
    """
    Charge credit-g (ID 31 sur OpenML) et applique un preprocessing standard.
    """
    print("--- Préparation des données Credit-g ---")
    data = fetch_openml(data_id=31, as_frame=True, parser='auto')
    df = data.frame

    # Target : 'good' -> 1, 'bad' -> 0
    y = (df['class'] == 'good').astype(int).values
    X = df.drop(columns=['class'])

    # Pipeline de prétraitement
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(
        include=['category', 'object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first',
             sparse_output=False), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)
    return train_test_split(X_processed, y, test_size=0.3, random_state=42)


def fit_base_logistic(X_train, y_train, X_test):
    """
    Entraîne le modèle de base (Régression Logistique) avant l'étude de l'incertitude.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def evaluate_base_model_credit(y_test, y_prob):
    """
    Évalue les performances de base du modèle de Régression Logistique.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC - Régression Logistique (Credit-g)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    return roc_auc


def bootstrap_confidence_intervals(X_train, y_train, X_test, n_bootstraps=100):
    """
    Estime la variance épistémique via Bootstrap.
    Renvoie la moyenne des probabilités et l'intervalle de confiance à 95%.
    """
    print("--- Exécution du Bootstrap (Calcul des IC) ---")
    n_samples = X_train.shape[0]
    preds = np.zeros((n_bootstraps, X_test.shape[0]))

    for i in range(n_bootstraps):
        # Tirage avec remise
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X_train[indices], y_train[indices]

        # Ajustement du modèle
        model = LogisticRegression(max_iter=1000)
        model.fit(X_boot, y_boot)
        preds[i, :] = model.predict_proba(X_test)[:, 1]

    return preds

    for alpha in alphas:
        lower_bound = 0.5 - alpha
        upper_bound = 0.5 + alpha

        # Filtre de la zone d'indécision
        decided_mask = (mean_prob < lower_bound) | (mean_prob > upper_bound)

        n_rejected = np.sum(~decided_mask)
        reject_rates.append(n_rejected / len(y_test))

        if np.sum(decided_mask) > 0:
            y_pred_decided = (mean_prob[decided_mask] > 0.5).astype(int)
            acc = np.mean(y_pred_decided == y_test[decided_mask])
            accuracies.append(acc)
        else:
            accuracies.append(np.nan)

    # Trace de la courbe
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, accuracies, label='Précision sur décidés',
             color='blue', marker='o')
    plt.plot(alphas, reject_rates, label='Taux de rejet (Incertitude)',
             color='red', linestyle='--')
    plt.xlabel('Seuil Alpha (Région indécision: [0.5-alpha, 0.5+alpha])')
    plt.ylabel('Taux')
    plt.title('Courbe de Rejet : Précision vs Taux d\'Incertitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def explore_ood_question(X_test, mean_prob, ci_lower, ci_upper):
    """
    Isole le point avec la plus grande incertitude épistémique (plus grand IC),
    analyse ses caractéristiques aberrantes, et introduit les concepts théoriques.
    """
    # Recherche du profil avec le plus grand désaccord entre les modèles Bootstrap
    ci_width = ci_upper - ci_lower
    idx_max_var = np.argmax(ci_width)

    print("\n" + "="*70)
    print("--- ÉTUDE DE CAS : ANALYSE DE L'INCERTITUDE EXTRÊME (OOD) ---")
    print("="*70)

    print(
        f"Le profil test n°{idx_max_var} génère un désaccord massif entre les modèles Bootstrap.")
    print(f"Probabilité moyenne calculée : {mean_prob[idx_max_var]:.2f}")
    print(
        f"Intervalle de Confiance (95%)  : [{ci_lower[idx_max_var]:.2f}, {ci_upper[idx_max_var]:.2f}]")
    print(f"Largeur totale de l'intervalle : {ci_width[idx_max_var]:.2f}\n")

    # Analyse des features de ce point (Recherche d'outliers)
    point_features = X_test[idx_max_var]
    # Les données numériques sont standardisées, on cherche les Z-scores > 2 ou < -2
    extremes = np.where(np.abs(point_features) > 2.0)[0]

    print("Analyse du profil (Variables standardisées) :")
    if len(extremes) > 0:
        print(
            f"Ce client possède des caractéristiques très atypiques (Z-score > 2) aux colonnes d'indices : {extremes}")
        print(
            f"Valeurs de ces caractéristiques : {np.round(point_features[extremes], 2)}")
        print("-> Il s'agit d'un point 'Out-Of-Distribution' (OOD) ou très rare.")
    else:
        print("Ce profil est atypique en raison d'une combinaison rare de ses variables catégorielles.")

    print("\n" + "-"*70)
    print("RÉPONSE ATTENDUE (Corrigé / Application Métier) :")
    print("La largeur de cet intervalle indique que l'IA ne sait pas évaluer ce client.")
    print("Dans un système de décision bancaire réel, ce client ne doit être ni accepté")
    print("ni rejeté automatiquement. C'est précisément l'Incertitude Épistémique qui")
    print("doit déclencher une 'Alerte' et le renvoi du dossier vers un expert humain.")
    print("="*70 + "\n")


def performance_incertitude(preds_boot, y_test):
    mean_prob = np.mean(preds_boot, axis=0)
    y_pred_final = (mean_prob >= 0.5).astype(int)
    # Plage des coefficients de confiance (de 40% à 98%)
    confidence_levels = np.linspace(0.40, 0.98, 50)

    accuracies_on_sure = []
    hesitation_rates = []

    for conf in confidence_levels:
        # Calcul des quantiles (ex: conf=0.90 -> q_low=5%, q_high=95%)
        alpha = 1.0 - conf
        q_low = (alpha / 2.0) * 100
        q_high = (1.0 - alpha / 2.0) * 100

        ci_lower = np.percentile(preds_boot, q=q_low, axis=0)
        ci_upper = np.percentile(preds_boot, q=q_high, axis=0)

        # Masque d'hésitation : le seuil de 0.5 est dans l'intervalle de confiance
        # C'est la fameuse "Région de prédiction / d'hésitation"
        hesitating_mask = (ci_lower < 0.5) & (ci_upper > 0.5)

        # Taux d'échantillons où le modèle hésite
        hesitation_rate = np.sum(hesitating_mask) / len(y_test)
        hesitation_rates.append(hesitation_rate)

        # Calcul de l'accuracy uniquement sur la partie "sûre" (~hesitating_mask)
        sure_mask = ~hesitating_mask
        if np.sum(sure_mask) > 0:
            acc = np.mean(y_pred_final[sure_mask] == y_test[sure_mask])
        else:
            acc = np.nan  # Si le modèle doute de TOUT, on ne peut pas calculer d'accuracy

        accuracies_on_sure.append(acc)

    # ==========================================
    #  AFFICHAGE DU GRAPHIQUE À DOUBLE AXE
    # ==========================================

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Axe de gauche : Précision (Accuracy)
    color1 = 'tab:blue'
    ax1.set_xlabel('Niveau de confiance exigé du Bootstrap')
    ax1.set_ylabel('Accuracy sur les prédictions "Sûres"', color=color1)
    ax1.plot(confidence_levels * 100, accuracies_on_sure,
             color=color1, lw=2, label="Accuracy (sur la partie sûre)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.5, 1.0)  # L'accuracy varie généralement entre 0.5 et 1.0

    # Axe de droite : Taux d'hésitation
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(
        'Pourcentage d\'échantillons rejetés (Hésitation)', color=color2)
    ax2.plot(confidence_levels * 100, np.array(hesitation_rates) * 100, color=color2,
             lw=2, linestyle='--', label="% d'Hésitation (Région de prédiction)")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)

    plt.title("Performance vs Incertitude Épistémique (Régions de prédiction)")
    fig.tight_layout()
    plt.show()


def plot_conformal_prediction_analysis(model, X_cal, y_cal, X_test, y_test):
    """
    Computes Split Conformal Prediction and plots Accuracy/Hesitation vs Coverage.
    """
    # 1. Calculate Conformity Scores on Calibration Set
    # Score = 1 - P(Y_true | X)
    probs_cal = model.predict_proba(X_cal)
    # Get the probability assigned to the actual correct class
    true_class_probs = probs_cal[np.arange(len(y_cal)), y_cal]
    scores = 1 - true_class_probs

    # 2. Test Set Probabilities
    probs_test = model.predict_proba(X_test)

    # 3. Iterate through different coverage levels (1 - alpha)
    coverages = np.linspace(0.70, 0.99, 30)
    empirical_accuracies = []
    hesitation_rates = []

    for target_cov in coverages:
        alpha = 1 - target_cov
        n = len(y_cal)
        # Quantile calculation (with finite sample correction)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(scores, q_level, method='higher')

        # Construct Prediction Sets
        # Class is in set if P(class|x) >= 1 - qhat
        set_0 = probs_test[:, 0] >= (1 - qhat)
        set_1 = probs_test[:, 1] >= (1 - qhat)

        # A prediction is "Correct" if the true label's class is in the set
        is_correct = []
        for i in range(len(y_test)):
            label = y_test[i]
            in_set = (label == 0 and set_0[i]) or (label == 1 and set_1[i])
            is_correct.append(in_set)

        # Accuracy = how often the TRUE label is inside the predicted SET
        empirical_accuracies.append(np.mean(is_correct))

        # Hesitation = how often the set contains BOTH {0, 1}
        hesitation = set_0 & set_1
        hesitation_rates.append(np.mean(hesitation))

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Empirical Accuracy (should follow the diagonal 1:1)
    ax1.plot(coverages * 100, empirical_accuracies, 'b-',
             lw=3, label='Précision réelle (Couverture)')
    ax1.plot([70, 100], [0.7, 1.0], 'k--', alpha=0.5,
             label='Ligne idéale (Garantie)')
    ax1.set_xlabel('Couverture Nominale Ciblée (1 - alpha) %')
    ax1.set_ylabel('Précision (La vérité est dans l\'ensemble)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0.65, 1.05)

    # Plot Hesitation Rate
    ax2 = ax1.twinx()
    ax2.plot(coverages * 100, np.array(hesitation_rates) * 100,
             'r--', lw=2, label='Taux d\'hésitation {0,1}')
    ax2.set_ylabel('Taux d\'hésitation (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 100)

    plt.title("Illustration de la Prédiction Conforme (Split Conformal)")
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.grid(alpha=0.3)
    plt.show()


# ==============================================================================
# PARTIE 3 : THÉORIE DE LA DÉCISION
# ==============================================================================


def optimize_threshold_credit(y_test, y_prob):
    """
    Trouve le seuil optimal via matrice de coût (Credit-g).
    Coût : Faux Positif (dit 'good' mais est 'bad') = 1
           Faux Négatif (dit 'bad' mais est 'good') = 5
    """
    print("\n--- Théorie de la Décision : Matrice de Coûts (Credit-g) ---")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    costs = []
    # Calcul manuel pour correspondre à l'intuition de la matrice
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        # y_test=0 (bad), y_test=1 (good)
        fp = np.sum((y_test == 0) & (y_pred_t == 1))
        fn = np.sum((y_test == 1) & (y_pred_t == 0))
        cost = fp * 1 + fn * 5
        costs.append(cost)

    opt_idx = np.argmin(costs)
    opt_threshold = thresholds[opt_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, costs, color='purple', linewidth=2)
    plt.axvline(opt_threshold, color='green', linestyle='--',
                label=f'Seuil Opt: {opt_threshold:.2f}')
    plt.axvline(0.5, color='gray', linestyle=':', label='Seuil standard: 0.5')
    plt.xlim(0, 1)
    plt.xlabel("Seuil de décision")
    plt.ylabel("Coût Total Espéré")
    plt.title("Optimisation du seuil selon la matrice de coût")
    plt.legend()
    plt.show()


def optimize_reject_kepler_multiple_costs(y_test, y_prob, cost_error=1.0):
    """
    Implémente l'option de rejet de Chow.
    Trace l'évolution du coût total pour différents prix de l'expert.
    Zone de rejet : [0.5 - theta, 0.5 + theta]
    """
    print("\n--- Théorie de la Décision : Délégation à l'Expert (Kepler) ---")
    thetas = np.linspace(0, 0.499, 50)
    # Différents tarifs horaires de l'astrophysicien !
    expert_costs = [0.1, 0.25, 0.4]

    plt.figure(figsize=(9, 5))
    colors = ['green', 'orange', 'red']

    for gamma, color in zip(expert_costs, colors):
        total_costs = []
        for theta in thetas:
            lower, upper = 0.5 - theta, 0.5 + theta

            # Statut: 1 si rejeté (délégué à l'expert)
            rejected = (y_prob >= lower) & (y_prob <= upper)

            # Prédiction de l'IA (sur les points décidés par la machine)
            y_pred = (y_prob > 0.5).astype(int)
            errors = (y_pred != y_test) & (~rejected)

            # Coût = (Erreurs IA * 1.0) + (Dossiers délégués * gamma)
            cost = np.sum(errors) * cost_error + np.sum(rejected) * gamma
            total_costs.append(cost)

        opt_idx = np.argmin(total_costs)
        opt_theta = thetas[opt_idx]

        plt.plot(thetas, total_costs, color=color, linewidth=2,
                 label=f'Coût Expert = {gamma} (Opt: $\\theta$={opt_theta:.2f})')
        plt.axvline(opt_theta, color=color, linestyle=':', alpha=0.7)

    plt.xlabel("Theta (Largeur de la zone d'indécision IA)")
    plt.ylabel("Coût Total (Erreurs IA + Honoraires Expert)")
    plt.title("Optimisation du système hybride IA/Humain selon le coût de l'Expert")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def optimize_reject_kepler(y_test, y_prob, cost_error=1.0, cost_expert=0.2):
    """
    Implémente l'option de rejet de Chow sur le dataset Kepler.
    Zone de rejet : [0.5 - theta, 0.5 + theta]
    """
    print("\n--- Théorie de la Décision : Option de Rejet (Kepler) ---")
    thetas = np.linspace(0, 0.5, 50)
    total_costs = []

    for theta in thetas:
        lower = 0.5 - theta
        upper = 0.5 + theta

        # Statut: 1 si rejeté (expert), 0 si décidé par IA
        rejected = (y_prob >= lower) & (y_prob <= upper)

        # Prédiction de l'IA (seulement sur les points non rejetés)
        y_pred = (y_prob > 0.5).astype(int)
        errors = (y_pred != y_test) & (~rejected)

        # Coût total = (nb erreurs IA * coût_erreur) + (nb rejets * coût_expert)
        cost = np.sum(errors) * cost_error + np.sum(rejected) * cost_expert
        total_costs.append(cost)

    opt_theta = thetas[np.argmin(total_costs)]

    plt.figure(figsize=(8, 4))
    plt.plot(thetas, total_costs, color='orange', linewidth=2)
    plt.axvline(opt_theta, color='red', linestyle='--',
                label=f'Theta Opt: {opt_theta:.2f}')
    plt.xlabel("Theta (Largeur zone de rejet)")
    plt.ylabel("Coût Total du système Hybride (IA + Expert)")
    plt.title(
        f"Optimisation de la délégation à l'expert (Coût expert = {cost_expert})")
    plt.legend()
    plt.show()


# ==============================================================================
# BLOC PRINCIPAL D'EXÉCUTION
# ==============================================================================
if __name__ == "__main__":

    # -- Exécution Partie 1 --
    try:
        # explorer_variables_kepler()
        df_kepler = load_and_prep_kepler()
        explore_and_check_assumptions(df_kepler)

        X_kep = df_kepler[['x1_eclipse', 'x2_duree', 'x3_temp']].values
        y_kep = df_kepler['target'].values

        # Split temporel/manuel non requis ici, on évalue sur le dataset entier pour le TP
        # (Pour gagner du temps, mais un split test est recommandable en exercice)
        params_mle = fit_mle_kepler(df_kepler)
        prior_E = df_kepler['target'].mean()
        prior_N = 1.0 - prior_E

        print(f"Prior Exoplanète P(E) : {prior_E:.3f}")
        print(f"Prior Anomalie P(N) : {prior_N:.3f}")
        y_prob_kep = predict_naive_bayes(X_kep, params_mle, prior_E)

        print("AUC Naïve Bayes (Kepler) :", auc(
            *roc_curve(y_kep, y_prob_kep)[:2]))
    except FileNotFoundError:
        print("Fichier 'cumulative.csv' non trouvé. Veuillez le télécharger depuis Kaggle.")

    # -- Exécution Partie 2 --
    X_train_cred, X_test_cred, y_train_cred, y_test_cred = load_and_prep_credit()

    # Modèle de base
    base_probs = fit_base_logistic(X_train_cred, y_train_cred, X_test_cred)
    evaluate_base_model_credit(y_test_cred, base_probs)

    # Incertitude Bootstrap
    preds_boot = bootstrap_confidence_intervals(
        X_train_cred, y_train_cred, X_test_cred)

    # Calcul des quantiles 2.5% et 97.5% pour l'IC à 95%
    ci_lower = np.percentile(preds_boot, 2.5, axis=0)
    ci_upper = np.percentile(preds_boot, 97.5, axis=0)
    mean_prob = np.mean(preds_boot, axis=0)

    explore_ood_question(X_test_cred, mean_prob, ci_lower, ci_upper)

    performance_incertitude(preds_boot, y_test_cred)

    X_train_prop, X_cal, y_train_prop, y_cal = train_test_split(
        X_train_cred, y_train_cred, test_size=0.2)
    model_cp = LogisticRegression().fit(X_train_prop, y_train_prop)
    plot_conformal_prediction_analysis(
        model_cp, X_cal, y_cal, X_test_cred, y_test_cred)

    # -- Exécution Partie 3 --
    # Matrice Credit
    optimize_threshold_credit(y_test_cred, base_probs)

    # Rejet Expert Kepler (On simule si le fichier a été chargé)
    if 'y_prob_kep' in locals():
        optimize_reject_kepler_multiple_costs(y_kep, y_prob_kep)
