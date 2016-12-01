import random
import numpy
import math
import pylab

Alpha = 0.1
Beta = 9
M_BASIS_FUNCTIONS = 9
N_TRAIN = 5
N_REAL = 1000

def main():
    x_real = numpy.linspace(0, 1, N_REAL)
    y_real = numpy.sin(2 * numpy.pi * x_real)
    x_train = numpy.linspace(0, 1, N_TRAIN)
    y_train = numpy.sin(2 * math.pi * x_train) + numpy.random.normal(0, 0.3, N_TRAIN)
    means = numpy.linspace(0, 1, M_BASIS_FUNCTIONS)

    m = mn(Alpha, Beta, means, 0.1, x_train, y_train)

    y_predict = [m.T.dot(phi(x, means, 0.1))[0, 0] for x in  x_real]
    print y_predict

    pylab.plot(x_train, y_train, 'bo')
    pylab.plot(x_real, y_real, 'g-')
    pylab.plot(x_real, y_predict, 'r-')
    pylab.show()

def basis_function(x, mean, scale):
    return math.exp(-(x - mean) ** 2 / (2 * scale ** 2))

def phi(x, means, scale):
    N = len(means)
    return numpy.array(
        [basis_function(x, m, scale) for m in means]).reshape(N, 1)

# return [train_x] * [means] matrix
def design_matrix(train_x, means, scale):
    N = len(means)
    res = None
    for x in train_x:
        if res is not None:
            res = numpy.vstack((res, phi(x, means, scale).reshape(1, N)))
        else:
            res = phi(x, means, scale).reshape(1, N)
    return res

def mn(alpha, beta, mean, scale, x_train, y_train):
    N = len(y_train)
    res = beta * numpy.linalg.inv(Sn_inv(x_train, alpha, beta, mean, scale))
    res = res.dot(design_matrix(x_train, mean, scale).T)
    res = res.dot(numpy.array(y_train).reshape(N, 1))
    return res

def Sn_inv(x_train, alpha, beta, mean, scale):
    N = len(mean)
    I = numpy.identity(N)
    dm = design_matrix(x_train, mean, scale)
    return alpha * I + beta * dm.T.dot(dm)


if __name__ == '__main__':
    main()
