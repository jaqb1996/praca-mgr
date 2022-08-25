import math
import numpy as np

class GH_Kalman_filter:
    def __init__(self, m, P, Q, R, processModel, measurementModel, degree, dt, kwargs=None):
        self.m = m
        self.P = P
        self.Q = Q
        self.R = R
        self.g = processModel
        self.h = measurementModel
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.p = degree
        self.dt = dt
        self.n = len(self.m)
        self.k = len(R)
        self.sigmaPoints, xis1d = self.__computeSigmaPoints()
        self.weights = self.__computeWeights(xis1d)
        self.pToNthPower = self.p ** self.n

    def predict(self, u):
        sqrtP = np.linalg.cholesky(self.P)
        sum_m = np.zeros((self.n, 1))
        sum_P = np.zeros((self.n, self.n))
        chis = []
        for i in range(self.pToNthPower):
            xi = self.sigmaPoints[i,:].reshape(self.n, 1)
            w = self.weights[i, 0]
            chi_prev = self.m + sqrtP.dot(xi)
            chi = self.g(chi_prev, u, self.dt, **self.kwargs)
            chis.append(chi)
            sum_m += w * chi
        self.m = sum_m
        for i in range(self.pToNthPower):
            w = self.weights[i, 0]
            chi = chis[i]
            sum_P += w * (chi - self.m).dot((chi - self.m).T)
        self.P = sum_P + self.Q

    def update(self, z):
        sqrtP = np.linalg.cholesky(self.P)
        sum_mu = np.zeros((self.k, 1))
        sum_S = np.zeros((self.k, self.k))
        sum_C = np.zeros((self.n, self.k))
        chi_bars = []
        ys = []
        for i in range(self.pToNthPower):
            xi = self.sigmaPoints[i,:].reshape(self.n, 1)
            w = self.weights[i, 0]
            chi_bar = self.m + sqrtP.dot(xi)
            chi_bars.append(chi_bar)
            y = self.h(chi_bar, **self.kwargs)
            ys.append(y)
            sum_mu += w * y
        mu = sum_mu
        for i in range(self.pToNthPower):
            w = self.weights[i, 0]
            chi_bar = chi_bars[i]
            y = ys[i]
            sum_S += w * (y - mu).dot((y - mu).T)
            sum_C += w * (chi_bar - self.m).dot((y - mu).T)
        S = sum_S + self.R
        C = sum_C
        K = C.dot(np.linalg.inv(S))
        self.m = self.m + K.dot(z - mu)
        self.P = self.P - K.dot(S).dot(K.T)
        
    def __computeSigmaPoints(self): 
        hermitePolynomial = self.__getHermitePolynomial(self.p)
        xis1d = np.roots(hermitePolynomial)
        E_next = xis1d
        for _ in range(self.n - 1):
            E_next = self.__starOperator(E_next, xis1d)
        if E_next.ndim < 2:
            E_next = E_next.reshape(len(xis1d), 1)
        return E_next, xis1d

    def __computeWeights(self, xis1d):
        lowerDegreeHermitePolynomial = self.__getHermitePolynomial(self.p - 1)
        Ws_1d = math.factorial(self.p)/(self.p**2 * np.polyval(lowerDegreeHermitePolynomial, xis1d)**2)
        w_next = Ws_1d
        for _ in range(self.n - 1):
            w_next = self.__crossOperator(Ws_1d, w_next)
        if w_next.ndim < 2:
            w_next = w_next.reshape(len(Ws_1d), 1)
        return w_next
        
    def __getHermitePolynomial(self, p):
        if p == 0:
            return np.array([1])
        if p == 1:
            return np.array([1, 0])
        return np.polysub(np.polymul([1, 0], self.__getHermitePolynomial(p - 1)), (p - 1) * self.__getHermitePolynomial(p - 2))

    def __starOperator(self, E, eta):
        if E.ndim < 2:
            E = E.reshape(len(eta), 1)
        nE, k = E.shape
        I_nE = np.ones((nE, 1))
        result = np.array([]).reshape(0, k + 1)
        for eta_i in eta:
            rows = np.hstack([E, eta_i * I_nE])
            result = np.vstack([result, rows])
        return result

    def __crossOperator(self, omega, w):
        if w.ndim < 2:
            w = w.reshape(len(omega), 1)
        result = np.array([]).reshape(0, 1)
        for omega_i in omega:
            result = np.vstack([result, omega_i * w])
        return result