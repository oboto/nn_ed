import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm

# s(0,y) = s(1,y) = s(x, 0) = 0
# ds(x, 1) = sen(pi*x)

#--------------hiperparametros----------------------
n_hidden = 3  # numero neuronas en la capa oculta
iter = 2  # numero de iteraciones sobre el conjunto de entrenamiento
lmb = 0.01  # tasa de aprendizaje
x0 = 0  # CI intervalo [0, 1]
x1 = 1  # fin del intervalo


# ecuacion diferencial parte derecha
def EcuacionDiferencial(x, dy):
    result = (2 - (np.pi**2) * (x[1]**2) * np.sin(np.pi * x[0]))
    return result


# solucion analitica
def sol_exacta(x):
    return x[1]**2 * np.sin(np.pi * x[0])


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


def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def derivada_salida_red_neuronal(W, x, k=1):
    # derivada de salida_red_neuronal np.dot(sigmoid(np.dot(x, W[0])), W[1])
    # usamos la regla de la cadena
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(np.dot(x, W[0])))


def B(x):
    return 2 * x[1] * np.sin(np.pi * x[0])


def salida_general(x, net_out, net_out1,  d_net_out):
    return B(x) + x[0] * (1 - x[0]) * x[1] * (net_out - net_out1 -d_net_out )

y_grad = grad(salida_general)
y_grad2 = grad(y_grad)

def loss_function(W, x, y):
    loss_sum = 0.

    for xi in x: # los nodos del intervalo que hemos escogido para x
        for yi in y: # los nodos del intervalo que hemos escogido para y
            input_point = np.array([xi, yi])
            # f(x, y)/dy
            dy = y_grad(xi, net_out)
            func = EcuacionDiferencial(input_point, dy)
            # N(x, y, p)
            net_out = salida_red_neuronal(W, input_point)[0]
            # N(x, 1, p)
            input_point1 = np.array([xi, 1])
            net_out1 = salida_red_neuronal(W, input_point1)[0]
            # N'(x, y, p)
            d_net_out = derivada_salida_red_neuronal(W, xi)[0][0]

            psy_t_hessian = jacobian(jacobian(salida_general))(input_point, net_out, net_out1, d_net_out)

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func) ** 2
            loss_sum += err_sqr
            print(loss_sum)
    return loss_sum


def sol_neuronal(W, x_space, lmb, iter):
    # descenso de gradiente estocástico ---------------------------------------------------------------
    for i in range(iter):  # entrenamiento iter veces
        loss_grad = grad(loss_function)(W, x_space, y_space)
        W[0] = W[0] - lmb * loss_grad[0]  # capa de entrada donde se aplica la funcion sigmoide
        W[1] = W[1] - lmb * loss_grad[1]  # capa de salida


def pintar_sol_neuronal(x_space, y_space, n,  W, ax):
    surface2 = np.zeros((n, n))
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            net_outt = salida_red_neuronal(W, [x, y])[0]
            surface2[i][j] = salida_general([x, y], net_outt)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x_space, y_space)
    surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    plt.title("NN-ECM: " + str(loss_function(W, x_space, y_space)))

    # error = 0
    # for i in range(len(x_space)):
    #     error += (res[i] - y_space[i]) ** 2
    # error /= len(x_space)
    #
    # print "Error: " + str(error)

def pintar_sol_analitica(x_space, y_space, n, ax):
    surface = np.zeros((n, n))
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            surface[i][j] = sol_exacta([x, y])

    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x_space, y_space)
    surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.title("Analitica")

# Datos ------------------------------------------------------------------------------
npr.seed(2)
n = 10  # numero de nodos en el intervalo
x_space = np.linspace(x0, x1, n)  # intervalo
y_space = np.linspace(x0, x1, n)
W = [npr.randn(2, 10), npr.randn(10, 1)]

sol_neuronal(W, x_space, lmb, iter)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
pintar_sol_neuronal(x_space, y_space, n, W, ax)

ax1 = fig.add_subplot(1, 2, 2, projection='3d')
pintar_sol_analitica(x_space, y_space, n, ax1)
plt.show()
