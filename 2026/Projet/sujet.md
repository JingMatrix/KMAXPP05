# KMAXPP05 — Projet Final de Machine Learning

Ce projet final pour les étudiants de L3 compte pour 8 points dans le cadre de l'évaluation CC2 (sur 20 points au total). Les étudiants doivent réaliser ce projet en groupes de 2, 3 ou 4. Vous devrez soumettre votre travail avant le jour de la soutenance et présenter votre projet lors de la dernière séance de TP, le _mardi 9 juin 2026_.

Plus de détails logistiques sont fournis à la fin de ce document.

## 1. Choix des Projets Disponibles
Trois sujets d'analyse de données s'offrent à vous. L'accès aux articles scientifiques de référence est garanti via le moteur de recherche de la bibliothèque universitaire : [Archipel - Univ Toulouse](https://catalogue-archipel.univ-toulouse.fr/primo-explore/search?vid=33UT3_VU1)

Veuillez choisir _un_ des jeux de données suivants :

_1. Reconnaissance d'Activité Humaine via Smartphones (HAR)_ `[Classification Multi-classes]`
*   _Contexte :_ Cette base de données a été construite à partir d'enregistrements de 30 participants effectuant des activités de la vie quotidienne (ADL) avec un smartphone fixé à la taille, équipé de capteurs inertiels. L'objectif est de classifier les données selon 6 activités (MARCHE, MONTÉE_ESCALIERS, DESCENTE_ESCALIERS, ASSIS, DEBOUT, ALLONGÉ).
*   _Données :_ [Lien Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
*   _Référence :_ [Article Springer](https://link.springer.com/chapter/10.1007/978-3-642-35395-6_30)

_2. Données de Supraconductivité_ `[Régression]`
*   _Contexte :_ L'objectif de ce projet est de prédire la température critique d'un supraconducteur à partir de 81 caractéristiques physiques extraites de sa formule chimique, telles que la conductivité thermique, le rayon atomique, la valence, l'affinité électronique et la masse atomique.
*   _Données :_ [Lien UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/464/superconductivty+data)
*   _Référence :_ [Prépublication ArXiv](https://arxiv.org/abs/1803.10260)

_3. Sélection de Candidats Pulsars (HTRU2)_ `[Classification Binaire Déséquilibrée]`
*   _Contexte :_ Les pulsars sont des types d'étoiles rares présentant un intérêt scientifique considérable. Les candidats collectés lors du relevé HTRU doivent être classifiés en classes "pulsar" ou "non-pulsar" pour faciliter de nouvelles découvertes. Attention : les vrais pulsars constituent une classe minoritaire !
*   _Données :_ [Lien UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/372/htru2)
*   _Référence :_ [Prépublication ArXiv](https://arxiv.org/abs/1603.05166)

## 2. Objectifs et Méthodologie
Votre objectif est d'accomplir la tâche de classification ou de régression spécifiée par le projet que vous avez choisi. Vous devez obligatoirement suivre ces trois étapes :

*   _Étape 1 : Exploration et Prétraitement des Données (EDA)._ Comprenez le contexte du projet. Vous devez traiter les valeurs manquantes et/ou normaliser les données si nécessaire, explorer la distribution des caractéristiques et sélectionner les variables pertinentes pour votre tâche de machine learning.
*   _Étape 2 : Machine Learning Classique._ Accomplissez la tâche en utilisant des outils de machine learning classiques abordés en cours ou en TP (ex. : Régression Linéaire/Logistique, Méthodes Bayésiennes, ACP/PCA, Arbres de Décision). Le choix du modèle peut s'inspirer des références fournies. _Note :_ Vous devez analyser vos résultats en utilisant des *métriques d'évaluation adaptées* à votre tâche spécifique.
*   _Étape 3 : Deep Learning._ Accomplissez la tâche avec un modèle de Réseau de Neurones, de préférence en utilisant TensorFlow. Vous devez vous assurer d'utiliser la bonne fonction de perte (*loss function*) et les couches d'activation finales appropriées à votre tâche. Le surapprentissage (*overfitting*) doit être traité, et des techniques de régularisation doivent être appliquées et expliquées.

## 3. Règles de Constitution des Groupes
Des TP1 à TP7, vous étiez répartis dans les _groupes de TP_ administratifs suivants :
*   _A11 :_ Jianyu Ma (jianyu.ma@math.univ-toulouse.fr)
*   _A12 :_ Marco Hanocq (marco.hanocq@irit.fr)
*   _A21 :_ Alexey Lazarev (alexey.lazarev@math.univ-toulouse.fr)

Pour ce projet final, vous formerez des _groupes de projet_ de 2, 3 ou 4 étudiants. Vous êtes autorisés à former des groupes de projet mixtes, c'est-à-dire en associant des étudiants issus de différents groupes de TP (ex. : un étudiant du A11 avec un étudiant du A21). En raison de cette flexibilité, nous deviendrons vos _encadrants de projet_. Chaque groupe de projet sera assigné à un encadrant, qui répondra à vos questions et fera partie du jury lors de votre soutenance.

Veuillez respecter les règles suivantes pour la formation de vos groupes :

1.  _Date limite de déclaration :_ Tous les groupes de projet doivent être fixés et confirmés **avant le lundi 1er juin 2026**. Vous devez envoyer un e-mail à l'un des enseignants pour déclarer votre équipe, en mettant _tous les membres en copie (CC)_ et en précisant clairement le groupe de TP d'origine (A11, A12 ou A21) de chaque membre. Merci de le faire le plus tôt possible.
2.  _Flexibilité des séances :_ Étant donné que les groupes de projet peuvent mélanger des étudiants de différents groupes de TP, vous êtes exceptionnellement autorisés à assister à la séance de TP d'un autre enseignant afin de pouvoir travailler physiquement avec les membres de votre groupe.
3.  _Affectation de l'encadrant :_ Un groupe de projet composé d'exactement 3 membres issus du *même* groupe de TP est automatiquement validé et conservera son enseignant d'origine comme encadrant de projet. À l'inverse, les groupes de 2 ou 4 membres, ou les groupes transverses (inter-TP), seront soumis à validation et à une éventuelle réaffectation de notre part afin d'équilibrer la charge de travail entre les trois encadrants.
4.  _Planning des soutenances :_ Une fois les groupes de projet et leurs encadrants validés, votre encadrant de projet vous enverra le planning des soutenances. Le jour de la soutenance (TP10), vous n'êtes autorisés à participer que sur le créneau horaire spécifiquement assigné à votre groupe de projet.

## 4. Soutenance, Soumission et Barème

_Soumission :_
Chaque groupe doit soumettre son Jupyter Notebook par e-mail à son encadrant de projet spécifique **avant le lundi 8 juin 2026 à 23h59**. 
*   *Exigence de reproductibilité :* Votre notebook doit s'exécuter proprement du début à la fin sans provoquer d'erreur. 

_Soutenance (Mardi 9 juin 2026) :_
*   Chaque groupe disposera d'un maximum de _15 minutes de présentation_, suivies de _5 minutes de questions-réponses_ avec le jury.
*   *Format de présentation :* La préparation d'un support visuel (ex. : PowerPoint, Beamer ou Google Slides) est _fortement recommandée_. Présenter directement en faisant défiler un Jupyter Notebook est souvent laborieux pour le groupe comme pour le jury, et rend la communication de vos résultats beaucoup plus difficile.

_Barème de notation (8 points au total) :_
La note sera attribuée de manière égale à chaque membre du groupe, sur la base du notebook soumis, de la présentation et de la phase de questions-réponses. Les points sont répartis comme suit :
*   _2 points :_ Exploration des données, prétraitement et explications (Étape 1).
*   _2 points :_ Implémentation en Machine Learning classique et analyse pertinente des métriques (Étape 2).
*   _2 points :_ Implémentation en Réseaux de Neurones et gestion du surapprentissage (Étape 3).
*   _2 points :_ Qualité globale de la présentation, des supports visuels, de l'accomplissement du projet et pertinence des réponses aux questions du jury.
