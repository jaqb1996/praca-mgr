import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, m, P, Q, R, processModel, measurementModel, F, H, dt, kwargs=None):
        self.m = m
        self.P = P
        self.R = R
        self.Q = Q
        self.g = processModel
        self.h = measurementModel
        self.F = F # Jacobian of process model
        self.H = H # Jacobian of measurement model
        self.dt = dt
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
    def predict(self, u):
        Fx = self.F(self.m, u, self.dt, **self.kwargs)
        self.P = Fx.dot(self.P).dot(Fx.T) + self.Q
        self.m = self.g(self.m, u, self.dt, **self.kwargs)
    def update(self, z):
        v = z - self.h(self.m, **self.kwargs)
        Hx = self.H(self.m, **self.kwargs)
        Hx_T = Hx.T
        S = Hx.dot(self.P).dot(Hx_T) + self.R
        K = self.P.dot(Hx_T).dot(np.linalg.inv(S))
        self.P = self.P - K.dot(S).dot(K.T)
        self.m = self.m + K.dot(v)
        
