import random
import numpy
import math
import pylab

N = 1000
Beta = 30.0
T0 = 0.5
T1 = 12
T2 = 0
T3 = 0
Sigma = 0.4

def main():
  x_train = [
    0.000000,
    0.111111,
    0.222222,
    0.333333,
    0.444444,
    0.555556,
    0.666667,
  ]

  y_train = [
    0.349486,
    0.830839,
    1.007332,
    0.971507,
    0.133066,
    0.166823,
    -0.848307,
  ]

  x_real = numpy.linspace(0, 1, N)
  y_real = numpy.sin(2 * math.pi * x_real)

  # kernel_func = gauss_kernel(Sigma)
  kernel_func = regression_kernel(T0, T1, T2, T3)

  y_predict = numpy.array([mean(x_train, y_train, x, kernel_func, Beta)[0] for x in  x_real])

  s = numpy.array([math.sqrt(s2(x_train, x, kernel_func, Beta)[0][0]) for x in x_real])
  lower = y_predict - 2 * s
  upper = y_predict + 2 * s

  pylab.plot(x_train, y_train, 'bo')
  pylab.plot(x_real, y_real, 'g-')
  pylab.plot(x_real, y_predict, 'r-')
  pylab.fill_between(x_real, lower, upper, color='pink')
  pylab.xlim(0.0, 1.0)
  pylab.ylim(-1.4, 1.4)
  pylab.show()

# 6.23
def gauss_kernel(sigma):
  return lambda x1, x2: math.exp(-(x1 - x2) ** 2 / (2 * sigma ** 2))

# 6.63
def regression_kernel(t0, t1, t2, t3):
  return lambda x1, x2: t0 * math.exp(- (t1 / 2) * (x1 - x2) ** 2) + t2 + t3 * x1 * x2

# 6.62
def Cn(xv, x, kernel, beta):
  N = len(xv)
  I = numpy.identity(N)
  return numpy.array(
      [kernel(xi, xj) for xi in xv for xj in xv]).reshape(N, N) + I / beta

def k(xv, x, kernel):
  N = len(xv)
  return numpy.array([kernel(xi, x) for xi in xv]).reshape(N, 1)

# 6.66
def mean(xv, tv, x, kernel, beta):
  kv = k(xv, x, kernel)
  cn = Cn(xv, x, kernel, beta)
  return kv.T.dot(numpy.linalg.inv(cn)).dot(tv)

# 6.67
def s2(xv, x, kernel, beta):
  c = kernel(x, x) + 1 / beta
  cn = Cn(xv, x, kernel, beta)
  kv = k(xv, x, kernel)
  return c - kv.T.dot(numpy.linalg.inv(cn)).dot(kv)

if __name__ == '__main__':
  main()
