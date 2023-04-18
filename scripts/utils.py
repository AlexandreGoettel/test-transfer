import numpy as np


def f(j, fmin, fmax, Jdes):
    g = np.log(fmax) - np.log(fmin)
    return fmin * np.exp(j*g / (Jdes - 1))


def r_prime(j, fmin, fmax, Jdes):
    g = np.log(fmax) - np.log(fmin)
    return f(j, fmin, fmax, Jdes) * (np.exp(g / (Jdes - 1)) - 1.)


def Jdes(fmin, fmax, resolution):
    g = np.log(fmax) - np.log(fmin)
    return int(np.floor(1 + g / np.log(1. + resolution)))


def N(j, fmin, fmax, fs, resolution):
    g = np.log(fmax) - np.log(fmin)
    J = Jdes(fmin, fmax, resolution)
    return fs/fmin * np.exp(-j*g / (J - 1.)) / (np.exp(g / (J - 1.)) - 1.)


def m(fmin, fmax, resolution):
    g = np.log(fmax) - np.log(fmin)
    J = Jdes(fmin, fmax, resolution)
    return 1. / (np.exp(g / (J - 1.)) - 1.)

def j(j0, J, g, epsilon, fmin, fmax, fs, resolution):
    Nj0 = N(j0, fmin, fmax, fs, resolution)
    return - (J - 1.) / g * np.log(Nj0*(1 - epsilon) * fmin/fs * (np.exp(g / (J - 1.)) - 1.))
