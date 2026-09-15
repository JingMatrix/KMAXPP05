#coding: utf-8
"""
Une implémentation complète d'un réseau de neurones utilisant juste numpy et illustrant le mode de fonctionnement des librairies industrielles
comme PyTorch ou TensorFlow.

Cédric Campguilhem, 2025-2026.
Pour le cours KMAXPP05 de l'Université de Paul Sabatier (Toulouse 3).
"""
import numpy as np


def is_tensor(obj):
    return hasattr(obj, "data")


class Tensor:
    """
    C'est la classe que l'on utilisera pour manipuler les tenseurs dans le réseau de neurones. Cette classe s'utilise en fait
    comme un array numpy.

    Mais, pour implémenter la rétro-propagation du gradient, on aura besoin plus tard d'avoir une classe dédiée pour stocker les gradients.
    """
    def __init__(self, array):
        """
        Constructeur.

        - array: valeur du tenseur (array numpy)
        """
        self.data = array

    @property
    def shape(self):
        """
        Retourne la 'shape' du tenseur.
        """
        return self.data.shape

    @property
    def size(self):
        """
        Retourne le nombre d'éléments dans le tenseur: sa taille.
        """
        return self.data.size

    @property
    def T(self):
        """
        Opération de transposition.
        """
        return Tensor(self.data.T)
    
    def __neg__(self):
        """
        Opération unaire de négation.
        """
        return Tensor(-self.data)
    
    def __add__(self, other):
        """
        Opération d'addition avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        if is_tensor(other):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        """
        Opération d'addition avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Opération de soustraction avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        if is_tensor(other):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        """
        Opération de soustraction avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        if is_tensor(other):
            return Tensor(other.data - self.data)
        return Tensor(other - self.data)

    def __mul__(self, other):
        """
        Opération de multiplication (membre à membre) avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        if is_tensor(other):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __rmul__(self, other):
        """
        Opération de multiplication avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Opération de division (membre à membre) avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        if is_tensor(other):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def __rtruediv__(self, other):
        """
        Opération de division avec un autre tenseur ou une constante.

        - other: autre tenseur ou constante
        """
        if is_tensor(other):
            return Tensor(other.data / self.data)
        else:
            return Tensor(other / self.data)
    
    def __matmul__(self, other):
        """
        Opération de multiplication matricielle.

        - other: autre tenseur
        """
        return Tensor(self.data @ other.data)

    def __pow__(self, other):
        """
        Opération de puissance avec un autre tenseur ou une constante.
        """
        if is_tensor(other):
            return Tensor(self.data ** other.data)
        else:
            return Tensor(self.data ** other)

    def __ge__(self, other):
        """
        Opération de comparaison >= avec un autre tenseur ou une constante.
        """
        if is_tensor(other):
            return Tensor(self.data >= other.data)
        else:
            return Tensor(self.data >= other)

    def __gt__(self, other):
        """
        Opération de comparaison > avec un autre tenseur ou une constante.
        """
        if is_tensor(other):
            return Tensor(self.data > other.data)
        else:
            return Tensor(self.data > other)

    def __iadd__(self, other):
        """
        Opération de modification en place += avec un autre tenseur ou une constante.
        """
        if is_tensor(other):
            self.data += other.data
        else:
            self.data += other
        return self

    def __isub__(self, other):
        """
        Opération de modification en place -= avec un autre tenseur ou une constante.
        """
        if is_tensor(other):
            self.data -= other.data
        else:
            self.data -= other
        return self

    def mean(self, axis=None):
        """
        Calcul de la valeur moyenne.
        """
        return Tensor(self.data.mean(axis=axis))

    def copy(self, data):
        """
        Copie les données fournies dans le tenseur.
        """
        if is_tensor(data):
            self.data[:] = data.data
        else:
            self.data[:] = data

    def __getitem__(self, *args):
        """
        Accès à des indices du tenseur
        """
        return Tensor(self.data.__getitem__(*args))

    def __str__(self):
        """
        Retourne une représentation textuelle du tenseur.
        """
        return str(self.data)

    def __repr__(self):
        """
        Retourne une représentation textuelle du tenseur.
        """
        return repr(self.data)

    def __array__(self, dtype=None, copy=None):
        """
        Permet la conversion automatique vers un array NumPy.
        """
        return np.asarray(self.data, dtype=dtype)

    def __float__(self):
        """Permet de faire float(tensor)"""
        return float(self.data.item())

    def __int__(self):
        """Permet de faire int(tensor)"""
        return int(self.data.item())

    def item(self):
        """La méthode 'à la PyTorch' pour récupérer le scalaire"""
        return self.data.item()

    def __len__(self):
        """Opérateur len()"""
        return self.shape[0]

    def reshape(self, *args, **kwargs):
        """
        Reshape le tableua numpy sous-jacent
        """
        self.data = self.data.reshape(*args, **kwargs)
        return self


class Parameter(Tensor):
    """
    Classe dérivée d'un Tensor qui permet de stocker un gradient pour la rétro-propagation du gradient.
    Les poids et biais de notre opérateur linéaire seront donc des Parameter et pas uniquement des Tensor.
    On utilisera les Tensor pour les entrées.
    """
    def __init__(self, array):
        """
        Constructeur.

        - array: valeur du tenseur (array numpy)        
        """
        super().__init__(array)  # Cela appelle le constructeur de la classe Tensor.
        self.grad = None


# Ici j'ai implémenté des méthodes de base de numpy mais pour la classe Tensor.
# Les fonctions logistic et softmax utilisent une variante qui permet de stabiliser numériquement le calcul.

def where(cond, x, y):
    if is_tensor(x):
        x = x.data
    if is_tensor(y):
        y = y.data
    if is_tensor(cond):
        cond = cond.data
    return Tensor(np.where(cond, x, y))


def exp(x):
    if is_tensor(x):
        x = x.data
    return Tensor(np.exp(x))


def tanh(x):
    if is_tensor(x):
        x = x.data
    return Tensor(np.tanh(x))


def log(x):
    if is_tensor(x):
        x = x.data
    return Tensor(np.log(x))


def abs(x):
    if is_tensor(x):
        x = x.data
    return Tensor(np.abs(x))


def sqrt(x):
    if is_tensor(x):
        x = x.data
    return Tensor(np.sqrt(x))


def ones(dim):
    return Tensor(np.ones(dim))


def ones_like(x):
    return Tensor(np.ones(x.shape))


def zeros(dim):
    return Tensor(np.zeros(dim))


def zeros_like(x):
    return Tensor(np.zeros(x.shape))


def maximum(x, y):
    if is_tensor(x):
        x = x.data
    if is_tensor(y):
        y = y.data
    return Tensor(np.maximum(x, y))


def max(x, axis=None, keepdims=False):
    if is_tensor(x):
        x = x.data
    return Tensor(np.max(x, axis=axis, keepdims=keepdims))


def sum(x, axis=None, keepdims=False):
    if is_tensor(x):
        x = x.data
    return Tensor(np.sum(x, axis=axis, keepdims=keepdims))


def logistic(x):
    # La formule "simple" de la fonction logistique est
    # Z = 1 / (1 + exp(-X))
    # Mais, si X est très petit (négatif), le terme en exponentielle devient très grand. On utilise une astuce
    # pour stabiliser numériquement le calcul qui repose sur la symétrie de la fonction vue en TD:
    # sigma(X) = 1 - sigma(-X)
    # Donc, pour X >= 0:
    # sigma(X) = 1 / (1 + exp(-X))
    # Sinon on calcule:
    # 1 - sigma(-X) qui est égal à exp(X) / (1 + exp(X))
    z = np.zeros_like(x.data)
    mask = x.data >= 0.
    exp_pos = np.exp(-x.data[mask])
    exp_neg = np.exp(x.data[~mask])
    z[mask] = 1 / (1 + exp_pos)
    z[~mask] = exp_neg / (1 + exp_neg)
    return Tensor(z)


def softmax(x):
    # On suppose ici que x est un tenseur de logits (n x K) avec n observations et K classes
    # La formule "simple" de la fonction softmax est:
    # softmax(x) = exp(x) / sum(exp(x))
    # On va utiliser une astuce pour limiter les erreurs de type overflow dans l'exponentielle
    # On calcule d'avord la valeur maximum de x pour chaque observation (le logit maximum pour chaque classe)
    # Le keepdims permet de garder un xmax de même forme que x
    xmax = max(x, axis=1, keepdims=True)
    
    # Au lieu de calculer exp(x), on calcule exp(x-xmax). Cela revient à diviser au numérateur et 
    # au dénominateur par exp(xmax):
    exp_x = exp(x - xmax)

    # Enfin on divise par la somme
    return exp_x / (sum(exp_x, axis=1, keepdims=True))
