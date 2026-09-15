#coding: utf-8
"""
Une implémentation complète d'un réseau de neurones utilisant juste numpy et illustrant le mode de fonctionnement des librairies industrielles
comme PyTorch ou TensorFlow.

Cédric Campguilhem, 2025-2026.
Pour le cours KMAXPP05 de l'Université de Paul Sabatier (Toulouse 3).
"""
import numpy as np
from tensor import *


#===============================================================================================================================#
# Classes de base pour définir notre réseau de neurones                                                                         #
#===============================================================================================================================#


class Module:
    """
    C'est une classe de base pour tous les opérateurs de notre réseau de neurones:
    - opérateur linéaire
    - fonction d'activation
    Elle nous permettra aussi de définir une architecture complète !

    Un module, de manière générale, permet de:
    - faire une passe avant
    - faire une passe arrière
    - initialiser les paramètres (poids et biais)
    - collecter l'ensemble des paramètres de l'opérateur ou du réseau de neurones
    """
    def __init__(self):
        """
        Constructeur.
        """
        self.input_cache = None  # pour sauvegarder les entrées de la passe avant pour la passe arrière
    
    def reset_parameters(self):
        """
        Initialisation aléatoire des paramètres. On considère que par défaut, il n'y en a pas.
        """
        pass

    def forward(self, X, Y=None):
        """
        Implémentation de la passe avant à partir des entrées.
        
        - X: entrées, de dimensions (n, m) avec n le nombre d'observations et m le nombre de caractéristiques
        - Y: entrée optionelle. Sera utilisé pour les fonctions de perte.
        """
        # On sauvegarde (met en cache) le tenseur passé à la méthode forward.
        self.input_cache = X, Y
        return X  # par défaut, on se contente de retourner le tenseur reçu en entrée

    def __call__(self, *args):
        """
        Raccourci pour réaliser une passe avant.
        """
        return self.forward(*args)

    def backward(self, grad=None):
        """
        Rétro-propage le gradient à partir du gradient passé en argument.

        - grad: gradient rétro-propagé
        """
        # On précise que cette méthode DOIT être définie pour chacun des modules, il n'y a pas de comportement par défaut
        raise NotImplementedError

    def parameters(self):
        """
        Retourne une liste contenant tous les paramètres 'apprenables' du module.
        """
        params = []
        # On parcourt tous les attributs de l'objet
        for key, value in self.__dict__.items():
            # Si l'attribut est un Paramètre, on l'ajoute à la liste
            if isinstance(value, Parameter):
                params.append(value)
            # Si l'attribut est lui-même un sous-module, on va chercher ses paramètres de façon récursive
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params


class Loss(Module):
    """
    Classe de base pour les fonctions de pertes. Elle ne fait rien de spécial.
    """
    pass


class Optimizer:
    def __init__(self, parameters, learning_rate=0.001, maximize=False):
        """
        Constructeur.

        - parameters: paramètres du réseau de neurones à optimiser
        - learning_rate: taux d'apprentissage
        - maximize: définit si on maximise ou minimise le critère
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.maximize = maximize

    def step(self):
        """
        Réalise un pas d'optimisation.
        """
        # On précise que cette méthode DOIT être définie pour chacun des modules, il n'y a pas de comportement par défaut
        raise NotImplementedError


#===============================================================================================================================#
# Définition de l'architecture d'un réseau dense (complètement connecté)                                                        #
#===============================================================================================================================#


class Sequential(Module):
    """
    C'est la classe qui va nous permettre de définir l'architecture de notre réseau de neurones comme une séquence d'opérateurs:
    - opérateur linéaire
    - fonction d'activation
    """
    def __init__(self, modules):
        """
        Constructeur.

        - modules: liste des modules (opérateurs) à exécuter en séquence.
        """
        self.modules = modules
        super().__init__()

    def reset_parameters(self):
        """
        Initialisation de tous les paramètres des modules de la séquence.
        """
        for module in self.modules:
            module.reset_parameters()

    def forward(self, X):
        """
        Implémentation de la passe avant à partir des entrées.
        
        - X: entrées, de dimensions (n, m) avec n le nombre d'observations et m le nombre de caractéristiques
        """
        for module in self.modules:
            X = module.forward(X)
        return X

    def backward(self, grad):
        """
        Rétro-propage le gradient à partir du gradient passé en argument.

        - grad: gradient rétro-propagé
        """
        for module in self.modules[-1::-1]:
            grad = module.backward(grad)
        return grad

    def parameters(self):
        """
        Retourne une liste contenant tous les paramètres apprenables de la séquence.
        """
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def __getitem__(self, key):
        """
        Operateur [] pour accéder à des modules de la séquence.
        """
        return self.modules[key]

        
#===============================================================================================================================#
# Opérateurs principaux pour un réseau dense (complètement connecté)                                                            #
#===============================================================================================================================#


class Logistic(Module):
    def forward(self, Z):
        """
        Implémentation de la passe avant à partir des entrées

        - Z: entrées, de dimensions (n, m) avec n le nombre d'observations et m le nombre de caractéristiques

        Retourne un tenseur de dimensions (n, out_features)
        """
        # On utilise une version stable de la fonction logistique (voir module nn.py)
        X = logistic(Z)
        
        # Il nous faut un appel explicite pour sauvegarder (mettre en cache) le résultat pour la rétro-propagation
        return super().forward(X)

    def backward(self, grad):
        """
        Rétro-propage le gradient à partir du gradient passé en argument.

        - grad: gradient rétro-propagé
        """
        # On récupère les entrées sauvegardées pour calculer le gradient
        # On avait stocké sigma(Z), soit X:
        X, _ = self.input_cache
        
        # On calcule le gradient de la fonction logistique:
        # sigma'(Z) = sigma(Z) * (1 - sigma(Z))
        _grad = X * (1 - X)
        
        # On propage. On fait juste un produit membre à membre (produit d'Hadamard):
        # de/dZ = grad * sigma'(Z)
        return grad * _grad


class Tanh(Module):
    def forward(self, Z):
        """
        Implémentation de la passe avant à partir des entrées

        - Z: entrées, de dimensions (n, m) avec n le nombre d'observations et m le nombre de caractéristiques

        Retourne un tenseur de dimensions (n, out_features)
        """
        # On calcule la tangente hyperbolique
        X = tanh(Z)
        
        # Il nous faut un appel explicite pour sauvegarder (mettre en cache) le résultat pour la rétro-propagation
        return super().forward(X)

    def backward(self, grad):
        """
        Rétro-propage le gradient à partir du gradient passé en argument.

        - grad: gradient rétro-propagé
        """
        # On récupère la sortie X = tanh(Z)
        X, _ = self.input_cache
        
        # Le gradient de tanh(Z) est simplement 1 - tanh(Z)**2
        # L'opération X**2 avec notre classe tenseur fait bien X * X membre à membre (produit d'Hadamard)
        # tanh'(Z) = 1 - tanh(Z)**2
        _grad = 1.0 - X**2
        
        # On propage. On fait juste un produit membre à membre (produit d'Hadamard):
        # de/dZ = grad * tanh'(Z)
        return grad * _grad


class ReLU(Module):
    def forward(self, Z):
        """
        Implémentation de la passe avant à partir des entrées

        - Z: entrées, de dimensions (n, m) avec n le nombre d'observations et m le nombre de caractéristiques

        Retourne un tenseur de dimensions (n, out_features)
        """
        # Il nous faut un appel explicite pour sauvegarder (mettre en cache) les entrées pour la rétro-propagation
        super().forward(Z)
        
        # La fonction suivante implémente la fonction indicatrice:
        # where(Z > 0, 1, 0)
        # et donc ReLU serait where(Z > 0, 1, 0) * Z, ce qui est équivalent au code ci-dessous:
        return where(Z > 0., Z, 0.)
        
    def backward(self, grad):
        """
        Rétro-propage le gradient à partir du gradient passé en argument.

        - grad: gradient rétro-propagé
        """
        # On récupère les entrées sauvegardées pour calculer le gradient
        Z, _ = self.input_cache

        # On calcule le gradient. Qui utilise aussi la fonction indicatrice:
        _grad = where(Z > 0., 1., 0.)
        
        # On propage. On fait juste un produit membre à membre (produit d'Hadamard):
        # de/dZ = grad * ReLU'(Z)
        return grad * _grad


class Softmax(Module):
    def forward(self, Z):
        """
        Implémente la passe avant du Softmax.
        
        - Z: tenseur des logits, de dimensions (n, out_features)
        """
        # On utilise une version stable de la fonction softmax (voir module nn.py) pour calculer les probabilités
        X = softmax(Z)

        # Il nous faut un appel explicite pour sauvegarder (mettre en cache) le résultat pour la rétro-propagation
        return super().forward(X)

    def backward(self, grad):
        """
        Rétro-propage le gradient à travers le Softmax.
        
        - grad: gradient rétro-propagé entrant, de dimensions (n, out_features)
        """
        # On récupère les probabilités calculées lors de la passe avant
        X, _ = self.input_cache

        # Je n'ai pas démontré cette formule dans le cours. Mais l'idée est la même:
        # On calcule le résultat de la contraction tensorielle entre le gradient qui est propagé avec le gradient de la softmax
        # La formule serait:
        # $\frac{\partial e}{\partial Z_j} = \hat{Y}_j \left( \delta_j - \sum_{\nu=1}^K \delta_{\nu} \hat{Y}_{\nu} \right)$
        # On calcule d'abord le terme de somme:
        S = sum(grad * X, axis=-1, keepdims=True)

        # On applique ensuite la formule générale vectorisée
        return X * (grad - S)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init="kaiming", gain=1.0):
        """
        Constructeur.

        Ceci est la fonction qui sera appelée lorsqu'un objet est crée. self est un argument muet qui permet d'utiliser l'objet depuis 
        les différentes méthodes que l'on implémentera.

        - in_features: nombre de caractéristiques d'entrées pour la couche
        - out_features: nombre de caractéristiques de sortie pour la couche (c'est aussi le nombre de neurones)
        - bias: inclusion du terme de biais
        - init: initialisation des poids ('kaiming' pour ReLU et 'xavier' pour Tanh et Logistic)
        - gain: facteur de gain pour l'initialisation (pour Tanh le gain est de 5/3, pour Logistic et ReLU, ce sera de 1).
        """
        # Cela permet de sauvegarder les arguments sur les nombres de caractéristiques directement dans l'objet, dans des attributs
        self.in_features = in_features
        self.out_features = out_features
        
        # Les paramètres (poids et biais) ne sont pas initialisés par défaut, leur valeur est indéfinie
        self.weight = None
        self.bias = None
        
        # On force une initialisation des paramètres
        self.init = init
        self.gain = gain
        self.use_bias = bias
        self.reset_parameters()
        
        # Gestion de l'héritage
        super().__init__()

    def reset_parameters(self):
        """
        Initialisation aléatoire des paramètres (poids et biais).
        
        Je n'ai implémenté ici que les variantes 'normales' des initialisations. On a vu en cours qu'elles donnaient des résultats 
        similaires au lois 'uniformes'.
        """
        # On commence par initialiser le terme de biais à zéro. C'est un vecteur ligne (1 x nb de sorties)
        if self.use_bias:
            # Dans ce cas, le biais va être optimisé donc on utilise un Parameter
            self.bias = Parameter(np.zeros((1, self.out_features)))
        else:
            # Dans ce cas, le biais va rester constant, on utilise un Tensor
            self.bias = Tensor(np.zeros((1, self.out_features)))

        # Pour le poids, on part d'une loi normale standard (moyenne = 0 et variance = 1):
        # Le poids est un tenseur de dimension (nb d'entrées x nb de sorties)
        W = np.random.randn(self.in_features, self.out_features)

        # C'est maintenant que l'on adapte le poids en fonction de l'initialisation choisie. Il suffit de multiplier
        # la valeur déjà échantillonée par la variance adaptée
        if self.init == "kaiming":
            # Initialisation de Kaiming He
            self.weight = Parameter(W * self.gain * np.sqrt(2 / self.in_features))
        else:
            # Initialisation de Xavier Glorot
            self.weight = Parameter(W * self.gain * np.sqrt(2 / (self.in_features + self.out_features)))

    def forward(self, X):
        """
        Implémentation de la passe avant à partir des entrées
        
        - X: entrées, de dimensions (n, in_features) avec n le nombre d'observations

        Retourne un tenseur de dimensions (n, out_features)
        """
        # Il nous faut un appel explicite pour sauvegarder les entrées pour la rétro-propagation
        super().forward(X)

        # Utilisation de l'opérateur linéaire: Z = X W + b
        return X @ self.weight + self.bias

    def backward(self, grad):
        """
        Rétro-propage le gradient à partir du gradient passé en argument.

        - grad: gradient rétro-propagé
        """
        # On récupère les entrées de la passe avant
        X, _ = self.input_cache
        n = X.shape[0]

        # On calcule et on stocke les gradients par rapport aux poids et biais
        # Pour les poids: de/dW = X^T delta
        self.weight.grad = X.T @ grad
        
        # Pour les biais: de/db = 1n^T delta
        if self.use_bias:
            self.bias.grad = ones((1, n)) @ grad
        
        # On calcule le gradient des erreurs par rapport aux entrées et on le propage
        # de/dX = delta W^T
        grad_X = grad @ self.weight.T
        return grad_X


#===============================================================================================================================#
# Fonctions de perte                                                                                                            #
#===============================================================================================================================#


class MSELoss(Loss):
    def forward(self, Yhat, Y):
        """
        Implémente une passe avant dans la fonction de perte.

        - Yhat: prédiction du réseau de neurones
        - Y: résultat attendu
        """
        # On calcule les résidus Y - Yhat
        e = (Y - Yhat)
        
        # On stocke les résidus pour la passe arrière
        super().forward(e)
        
        # On calcule l'erreur quadratique
        se = e**2
        
        # On réduit pour garder l'erreur quadratique moyenne
        # En effet l'erreur quadratique moyenne est juste (1 / n) sum_i (y_i - yhat_i)**2
        return se.mean()

    def backward(self):
        """
        Initie la rétro-propagation du gradient.      
        """
        # On récupère les erreurs
        e, _ = self.input_cache
        n = e.size
        
        # On calcule le gradient et on propage
        # de/dYhat = (2 / n) * (Yhat - Y)
        return -2 / n * e


class BCEWithLogitsLoss(Loss):
    def forward(self, Z, Y):
        """
        Calcule la perte d'entropie croisée binaire à partir des logits.
        
        - Z: les logits sortant de la dernière couche linéaire
        - Y: les vraies étiquettes, contenant des 0 ou des 1
        """        
        # Calcul de la probabilité P = Sigmoid(Z) (version numériquement stable)
        Yhat = logistic(Z)
        
        # On sauvegarde les probabilités et les cibles pour la passe arrière
        super().forward(Yhat, Y)

        # Calcul des pertes (avec astuce de stabilité numérique)
        # On ajoute un minuscule epsilon (1e-15) pour éviter que log(0) ne fasse exploser le code
        eps = 1e-15
        e = - Y * log(Yhat + eps) - (1 - Y) * log(1 - Yhat + eps)
        
        # On retourne la moyenne
        return e.mean()

    def backward(self, grad=None):
        """
        Initie la rétro-propagation du gradient.
        """
        # On récupère Yhat et Y
        Yhat, Y = self.input_cache
        n = Yhat.shape[0]
        
        # Le gradient par rapport aux logits est: de/dZ = (Yhat - Y) / n
        return (Yhat - Y) / n


class CrossEntropyLoss(Loss):
    def forward(self, Z, Y):
        """
        Calcule la perte d'entropie croisée à K classes à partir des logits purs.
        
        - Yhat: les logits sortant de la dernière couche linéaire, dimensions (n, K)
        - Y: les vraies étiquettes en format ONE-HOT, dimensions (n, K)
        """
        # Calcul de la probabilité P = Softmax(Z) (version numériquement stable)
        Yhat = softmax(Z)
        
        # On sauvegarde les probabilités et les cibles pour la passe arrière
        super().forward(Yhat, Y)
        
        # Calcul de la perte (avec astuce de stabilité numérique)
        # On ajoute un minuscule epsilon (1e-15) pour éviter que log(0) ne fasse exploser le code
        # On multiplie terme à terme, on somme sur les colonnes (classes), puis on fait la moyenne du batch
        eps = 1e-15
        e = -sum(Y * log(Yhat + eps), axis=1)
        
        # On retourne la moyenne
        return e.mean()

    def backward(self, grad=None):
        """
        Initie la rétro-propagation du gradient.
        """
        # On récupère Yhat et Y
        Yhat, Y = self.input_cache
        n = Yhat.shape[0] 
        
        # Le gradient par rapport aux logits est: de/dZ = (Yhat - Y) / n
        return (Yhat - Y) / n
        

#===============================================================================================================================#
# Optimiseurs                                                                                                                   #
#===============================================================================================================================#


class SGD(Optimizer):
    """
    Algorithme de la descente de gradient 'simple'.
    """
    def step(self):
        """
        Réalise un pas d'optimisation.
        """
        # On itère sur l'ensemble des paramètres
        for param in self.parameters:
            
            # Dans le cas d'une maximisation, montée de gradient
            # P <- P + alpha * grad
            if self.maximize:
                param += self.learning_rate * param.grad
            
            # Dans le cas d'une minimisation, descente de gradient
            # P <- P - alpha * grad
            else:
                param -= self.learning_rate * param.grad


class SGDWithMomentum(Optimizer):
    """
    Algorithme de la descente de gradient utilisant la technique du momentum pour le lissage de la trajectoire.
    """
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, maximize=False):
        """
        Constructeur.

        - parameters: paramètres du réseau de neurones à optimiser
        - learning_rate: taux d'apprentissage
        - beta1: coefficient de friction, une valeur entre 0.8 et 0.99 est typique.
        - maximize: définit si on maximise ou minimise le critère
        """        
        # Gestion de l'héritage
        super().__init__(parameters, learning_rate, maximize)
        
        # Paramètre pour le momentum
        self.beta1 = beta1

        # Initialisation de l'historique des gradients pour le lissage
        self.gradients = {p: zeros_like(p) for p in parameters}

    def step(self):
        """
        Réalise un pas d'optimisation.
        """
        # On itère sur l'ensemble des paramètres
        for param in self.parameters:
            
            # Lissage par moyenne à exponentielles pondérées -> termes de gradient
            # v <- (1 - beta1) * grad + beta1 * v
            vold = self.gradients[param]
            vnew = (1 - self.beta1) * param.grad + self.beta1 * vold
            self.gradients[param] = vnew
            
            # Dans le cas d'une maximisation, montée de gradient
            # P <- P + alpha * v
            if self.maximize:
                param += self.learning_rate * vnew
        
            # Dans le cas d'une minimisation, descent de gradient
            # P <- P - alpha * v
            else:
                param -= self.learning_rate * vnew


class RMSprop(Optimizer):
    """
    Algorithme de la descente de gradient utilisant la technique du taux d'apprentissage (vitesse) adaptatif.
    """
    def __init__(self, parameters, learning_rate=0.001, beta2=0.9, maximize=False):
        """
        Constructeur.

        - parameters: paramètres du réseau de neurones à optimiser
        - learning_rate: taux d'apprentissage
        - beta2: coefficient de friction, une valeur entre 0.8 et 0.99 est typique.
        - maximize: définit si on maximise ou minimise le critère
        """        
        # Gestion de l'héritage
        super().__init__(parameters, learning_rate, maximize)

        # Paramètres pour RMSprop
        self.beta2 = beta2
        self.eps = 1e-8

        # Initialisation de l'historique des gradients au carré pour le lissage
        self.squaredgrad = {p: zeros_like(p) for p in parameters}
        
    def step(self):
        """
        Réalise un pas d'optimisation.
        """
        # On itère sur l'ensemble des paramètres
        for param in self.parameters:
            
            # Lissage par moyenne à exponentielles pondérées -> termes de gradient au carré (pour la vitesse)
            # s <- (1 - beta2) * grad**2 + beta2 * s  
            sqgradold = self.squaredgrad[param]
            sqgradnew = (1 - self.beta2) * param.grad**2 +  self.beta2 * sqgradold
            self.squaredgrad[param] = sqgradnew
            
            # Dans le cas d'une maximisation, montée de gradient
            # P <- P + alpha / (sqrt(sp) + eps) * grad
            if self.maximize:
                param += self.learning_rate / sqrt(sqgradnew + self.eps) * param.grad
            
            # Dans le cas d'une minimisation, descent de gradient
            # P <- P - alpha / (sqrt(sp) + eps) * grad
            else:
                param -= self.learning_rate / sqrt(sqgradnew + self.eps) * param.grad


class Adam(Optimizer):
    """
    Algorithme de la descente de gradient avec momentum et taux d'apprentissage adaptatif.
    """
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, maximize=False):
        """
        Constructeur.

        - parameters: paramètres du réseau de neurones à optimiser
        - learning_rate: taux d'apprentissage
        - beta1: coefficient de friction, une valeur entre 0.8 et 0.99 est typique.
        - beta2: coefficient de friction, une valeur entre 0.8 et 0.99 est typique.
        - maximize: définit si on maximise ou minimise le critère
        """        
        # Gestion de l'héritage
        super().__init__(parameters, learning_rate, maximize)
        
        # Paramètres pour RMSprop
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        
        # Initiation des termes d'historique pour les gradients, les gradients au carré 
        self.gradients = {p: zeros_like(p) for p in parameters}
        self.squaredgrad = {p: zeros_like(p) for p in parameters}
        
        # Initialisation du paramètre pour la correction de biais
        self.t = 0

    def step(self):
        """
        Réalise un pas d'optimisation.
        """
        # On met à jour les itérations
        self.t += 1
        
        # On itère sur l'ensemble des paramètres
        for param in self.parameters:
            
            # Lissage par moyenne à exponentielles pondérées avec correction de biais -> termes de gradient
            # v <- (1 - beta1) * grad + beta1 * v
            vold = self.gradients[param]
            vnew = (1 - self.beta1) * param.grad + self.beta1 * vold
            self.gradients[param] = vnew
            
            # Correction de biais: vhat <- v / (1 - beta1**t)
            vnew = vnew / (1 - self.beta1**self.t)

            # Lissage par moyenne à exponentielles pondérées avec correction de biais -> termes de gradient au carré (pour la vitesse)
            # s <- (1 - beta2) * grad**2 + beta2 * s             
            sqgradold = self.squaredgrad[param]
            sqgradnew = (1 - self.beta2) * param.grad**2 + self.beta2 * sqgradold
            self.squaredgrad[param] = sqgradnew

            # Correction de biais: : shat <- s / (1 - beta2**t)
            sqgradnew = sqgradnew / (1 - self.beta2**self.t)  
            
            # Dans le cas d'une maximisation, montée de gradient
            # P <- P + alpha / (sqrt(shat) + eps) * vhat
            if self.maximize:
                param += self.learning_rate / sqrt(sqgradnew + self.eps) * vnew 
            # Dans le cas d'une minimisation, descent de gradient
            # P <- P - alpha / (sqrt(shat) + eps) * vhat
            else:
                param -= self.learning_rate / sqrt(sqgradnew + self.eps) * vnew 
