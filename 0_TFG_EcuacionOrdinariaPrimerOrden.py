import numpy as np
import autograd.numpy as np
import sympy
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

# y'= f(x, y) = 2x - 1
# y(x0) = y0  -> y(0) = - 2
#--------------hiperparametros----------------------
n_hidden = 3  # numero neuronas en la capa oculta
iter = 100  # numero de iteraciones sobre el conjunto de entrenamiento
lmb = 0.01  # tasa de aprendizaje
x0 = 0  # CI intervalo [0, 1]
x1 = 1  # fin del intervalo
y0 = -2 # condicion inicial en el origen


# ecuacion diferencial parte derecha
def EcuacionDiferencial(x, y):
    return 2*x - 1

# solucion analitica
def sol_exacta(x):
    return x**2 - x - 2


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


def loss_function(W, x, y0):
    loss_sum = 0.
    for xi in x:  # los nodos del intervalo que hemos escogido
        # la salida del punto xi que da la red neuronal una vez
        # que ha pasado por las dos capas. Sabiendo que a la última
        # capa no se le aplica la función sigmoide
        net_out = salida_red_neuronal(W, xi)[0][0]
        # ecuacion general propuesta y = A + x*N(x,p)
        salidaY = y0 + xi * net_out
        # ecuacion diferencial f(x, y) = 2*x - 1  en el punto xi y en la salida hayada con
        # la ecuacion general
        func = EcuacionDiferencial(xi, salidaY)

        # derivada de la salida neuronal, se usa la regla de la cadena
        d_net_out = derivada_salida_red_neuronal(W, xi)[0][0]
        # la derivada de la ecuacion general y = A + x*N(x,p)
        d_solucion_general = net_out + xi * d_net_out

        # error cuadratico medio
        err_sqr = (d_solucion_general - func) ** 2
        # acumulacion del error cuadratico medio
        loss_sum += err_sqr
    return loss_sum


def sol_neuronal(W, x_space, lmb, iter, y0):
    # descenso de gradiente estocástico ---------------------------------------------------------------
    for i in range(iter):  # entrenamiento iter veces
        loss_grad = grad(loss_function)(W, x_space, y0)
        W[0] = W[0] - lmb * loss_grad[0]  # capa de entrada donde se aplica la funcion sigmoide
        W[1] = W[1] - lmb * loss_grad[1]  # capa de salida
        ecm = loss_function(W, x_space, y0)


def pintar_sol_neuronal(y0, x_space, W, ax):
    # dibujo solucion neuronal---------------------------------------------------------------------------
    ecm = loss_function(W, x_space, y0)
    res = [y0 + xi * salida_red_neuronal(W, xi)[0][0] for xi in x_space]
    ax.plot(x_space, res, 'r', label="Sol. NN")
    # ax.legend()


def pintar_sol_analitica(x_space, ax):
    # solucion analitica---------------------------------------------------------------------------------
    y_space = sol_exacta(x_space)
    ax.plot(x_space, y_space, 'g', label="Sol. Analitica")


i = 0  # fila
j = 0  # columna
np.random.seed(3)
fig, ax = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
nx = 25  # numero de cortes
x = sympy.Symbol('x')
x_space = np.linspace(x0, x1, nx)  # intervalo

while n_hidden < 10:
    while iter < 301:
        # hiperparametros----------------------------------------------------------
        W = [npr.randn(1, n_hidden), npr.randn(n_hidden, 1)]  # inicializacion de pesos aleatorios de una distribución normal
        sol_neuronal(W, x_space, lmb, iter, y0)
        pintar_sol_analitica(x_space, ax[i, j])
        pintar_sol_neuronal(y0, x_space, W, ax[i, j])
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
