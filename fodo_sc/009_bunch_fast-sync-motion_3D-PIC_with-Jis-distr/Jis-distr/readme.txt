Please use the first 6 columns that corresponds to
(x,gamma_betaX,y,gamma_betaY,phi,-delta gamma),
where phi is based 65MHz reference frequency, x and y
are normalized by a scaling constant (c/omega = 0.734053204259647).

PyHT x = 0.734053204259647 * x
PyHT x' = gamma_betaX / (beta * gamma)
PyHT z = - beta * c * (phi / (2 * pi * 65 MHz))
-delta gamma := gamma0 - gamma_particles 
PyHT dp = - (-delta gamma) / (beta**2 * gamma)
