import numpy as np
import autograd.numpy as np
import sympy
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt


#--------------hiperparametros----------------------
n_hidden = 3  # numero neuronas en la capa oculta
iter = 100  # numero de iteraciones sobre el conjunto de entrenamiento
lmb = 0.001  # tasa de aprendizaje
x0 = 0  # CI intervalo [0, 2]
x1 = 2  # fin del intervalo


# ecuacion diferencial parte derecha
def EcuacionDiferencial(x, y, dy):
    return -1./5. * np.exp(-x/5.) * np.cos(x) - 1./5. * dy - y


# solucion analitica
def sol_exacta(x):
    return np.exp(-x/5.) * np.sin(x)


def sigmoid(x):
    # recibe un valor x y devuelve un valor entre 0 y 1. Indica la probabilidad de un estado.
    return 1. / (1. + np.exp(-x))

def sigmoid_grad(x):
    # derivada del sigmoid
    return (1.0 - sigmoid(x)) * sigmoid(x)

def salida_red_neuronal(W, x):
    # la salida del punto xi que da la red neuronal una vez
    # que ha pasado por las dos capas
    # dot = producto matricial
    a1 = sigmoid(np.dot(x, W[0]))  #  resultados de la capa oculta
    # en la ultima capa no se aplica la funcion sigmoide
    return np.dot(a1, W[1])  # la salida de la capa oculta pasa por la capa de salida

def derivada_salida_red_neuronal(W, x, k=1):
    # derivada de salida_red_neuronal np.dot(sigmoid(np.dot(x, W[0])), W[1])
    # usamos la regla de la cadena
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(np.dot(x, W[0])))

def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def salida_general(xi, net_out):
    return xi + xi ** 2 * net_out


y_grad = grad(salida_general)
y_grad2 = grad(y_grad)


def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = salida_red_neuronal(W, xi)[0][0]
        # ecuacion general y = A + A'x + N(x,p)*x**2  --> A =y(o)=0  A'=y'(0) = 1
        salidaY = salida_general(xi, net_out)
        dy = y_grad(xi, net_out)
        # ecuacion diferencial f(x, y) en el punto xi y en la salida hayada con
        # la ecuacion general
        func = EcuacionDiferencial(xi, salidaY, dy)

        second_gradient_of_trial = y_grad2(xi, net_out)

        err_sqr = (second_gradient_of_trial - func) ** 2
        loss_sum += err_sqr
    return loss_sum


def sol_neuronal(W, x_space, lmb, iter):
    # descenso de gradiente estocástico ---------------------------------------------------------------
    for i in range(iter):  # entrenamiento iter veces
        loss_grad = grad(loss_function)(W, x_space)
        W[0] = W[0] - lmb * loss_grad[0]  # capa de entrada donde se aplica la funcion sigmoide
        W[1] = W[1] - lmb * loss_grad[1]  # capa de salida


def pintar_sol_neuronal(x_space, W, ax):
    # dibujo solucion neuronal---------------------------------------------------------------------------
    # ecm = loss_function(W, x_space)
    res = [salida_general(xi, salida_red_neuronal(W, xi)[0][0]) for xi in x_space]
    ax.plot(x_space, res, 'r', label="Sol. NN")


def pintar_sol_analitica(x_space, ax):
    # solucion analitica---------------------------------------------------------------------------------
    y_space = sol_exacta(x_space)
    ax.plot(x_space, y_space, 'g', label="Sol. Analitica")


i = 0  # fila
j = 0  # columna
npr.seed(2)
fig, ax = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
nx = 10  # numero de cortes
x_space = np.linspace(x0, x1, nx)  # intervalo

while n_hidden < 10:
    while iter < 301:
        W = [npr.randn(1, n_hidden), npr.randn(n_hidden, 1)]  # inicializacion de pesos aleatorios de una distribución normal
        sol_neuronal(W, x_space, lmb, iter)
        pintar_sol_analitica(x_space, ax[i, j])
        pintar_sol_neuronal(x_space, W, ax[i, j])
        iter += 100
        j += 1
    if i == 0:
        ax[i, 0].set_title("100 iteraciones")
        ax[i, 1].set_title("200 iteraciones")
        ax[i, 2].set_title("300 iteraciones")
    ax[i, 0].set_ylabel(str(n_hidden) + " neuronas")
    iter = 100
    j = 0
    i += 1
    n_hidden += 3
plt.suptitle("verde-solucion analitica  rojo- solucion NN")
plt.show()