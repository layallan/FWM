from sympy import *
import matplotlib.pyplot as plt
import math
import numpy as np

sl = 2.998 * 10 ** 8

a = 3.06 * 10**-44 #edit to adjust
b = - 3.67 * 10**-28 #edit to adjust
c = 9.11 * 10**-13 #edit to adjust
d = -644.07 #edit to adjust

omega = Symbol('omega')
phiIR = Symbol('phiNIR')

omegaIR = 1.57 * 10**15
omegaSHG = 2 * omegaIR
omegafwm = 2 * omegaSHG - omegaIR
phiIR = a * omega**3 + b * omega**2 + c * omega + d

T0 = phiIR.subs(omega, omegaIR)
d1 = diff(phiIR, omega)
T1 = d1.subs(omega, omegaIR)
d2 = diff(d1, omega)
T2 = d2.subs(omega, omegaIR)
d3 = diff(d2, omega)
T3 = d3.subs(omega, omegaIR)

P1 = (omega - omegafwm) * T1
P2 = 0.5 * ((omega - omegafwm)**2) * T2
P3 = ((omega - omegafwm)**3) * T3 / 6
phiUV = T0 - P1 + P2 - P3


xpointsIR = np.linspace(1.3*10**15, 1.8*10**15, num=15) #can edit the range here
xpointsIR = np.linspace(1.3*10**15, 9.42*10**15, num=15) #can edit the range here

exprsIR = []
for i in range(15):
    exprsIR.append(a*omega**3+b*omega**2+c*omega+d)
ypointsIR = [[expr.subs({omega:val}) for expr in exprsIR] for val in xpointsIR]
plt.figure()
plt.plot(xpointsIR*1e-15, ypointsIR)
plt.xlabel("Angular Frequency (rad/fs)")
plt.ylabel("Spectral Phase")
plt.title("IR Spectral Phase")
#plt.show() #shows initial IR plot in terms of angular frequency

xpointsUV = np.linspace(1.3*10**15, 8.2*10**15, num=15) #can edit the range here (same range as above)
exprsUV = []
for i in range(15):
    exprsUV.append(T0 - P1 + P2 - P3)
ypointsUV = [[expr.subs({omega: val}) for expr in exprsUV] for val in xpointsUV]
plt.figure()
plt.plot(xpointsUV*1e-15, ypointsUV)
plt.xlabel("Angular Frequency (rad/fs)")
plt.ylabel("Spectral Phase")
plt.title("UV Spectral Phase")
#plt.show() #shows resulting UV plot in terms of angular frequency

xpointsIRW = [2 * math.pi * sl * 10**9 / val for val in xpointsIR] #automatically changes the range as you edited above
plt.figure()
plt.plot(xpointsIRW, ypointsIR)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Phase")
plt.title("IR Spectral Phase")
#plt.show() #shows initial IR plot in terms of wavelength

xpointsUVW =  [2 * math.pi * sl * 10**9 / val for val in xpointsUV] #automatically changes the range as you edited above
plt.figure()
plt.plot(xpointsUVW, ypointsUV)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Phase")
plt.title("UV Spectral Phase")
#plt.show() #shows resulting UV plot in terms of wavelength

plt.show() #shows all plots
