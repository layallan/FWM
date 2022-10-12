# -*- coding: utf-8 -*-
import wave
import numpy as np
import sys
#from pyrsistent import v
from scipy import pi
import matplotlib.pyplot as plt
import math
import sympy as sp
import time
from FWM_transferred import A
eps=np.finfo(float).eps # smallest number
np.set_printoptions(threshold=sys.maxsize)

'''
Constants
'''
c = 299792458.0 # m/s
mu_0 = 4.0*pi*1e-7 # N/A**2
eps0 = 1.0/(mu_0*c**2) # F/m
h = 6.62606896e-34 # J s
hbar = h/(2.0*pi) # J s
e = 1.602176487e-19 # C
m_e = 9.10938215e-31 # kg
NA = 6.02214179e23 # 1/mol
kb = 1.3806504e-23 # J/K
a_B = 0.52917720859e-10 # m
bar = 1e5 # Pa
atm = 1.01325e5 # Pa 


'''
DISPERSION
'''
"""
Function def: rN_gas
    Helper function for equations-just makes equations less bulky
    returns constant
"""
def rN_gas(p, T): # for moderate pressure/Temperature(e.g. <50bar), for higher pressure need better compressibility model (non-ideal gas)
    T0 = 273.15
    p0 = 1.01325
    return p/p0*T0/T

# dispersion function

"""
Function def: d_gas
    returns refractive index of input gas for each respective frequency input
    Note the refractive index for the PCF and HCF is different from the refractive index of the gas
"""
def d_gas(w, gas, p=1.0, T=293.15): 
    """ refractive index of a 'gas'
        wavelength: wl [m]
        gas: gas [string]
        pressure: P [bar]
        temperature: T [K]
    """  
    rN = rN_gas(p, T)
    #E. R. Peck and S. Huang, "Refractivity and dispersion of hydrogen in the
    # visible and near infrared," J. Opt. Soc. Am. 67, 1550-1554 (1977). 
    if gas == 'He':
        B1, C1, B2, C2 = 4977.77, 28.54, 1856.94, 7.760
    elif gas == 'Ne':
        B1, C1, B2, C2 = 9154.48, 656.97, 4018.63, 5.728
    elif gas == 'Ar':
        B1, C1, B2, C2 = 20332.29, 206.12, 34458.31, 8.066
    elif gas == 'Kr':
        B1, C1, B2, C2 = 26102.88, 2.01, 56946.82, 10.043
    elif gas == 'Xe':
        B1, C1, B2, C2 = 103701.61, 12.75e3, 31228.61,  0.561
    elif gas == 'Air':
        B1, C1, B2, C2 = 14926.44, 19.36, 41807.57, 7.434
    else:
        raise ValueError('gas unknown')

    #assume dispersion is unitless? check?
    wl =2*pi*c/w # wavelength uncessary calculation just for clarity
    wl2 = (wl*1e6)**2 # usal spectroscopist units
    return rN*(B1*1e-8*wl2/(wl2 - C1*1e-6)+ B2*1e-8*wl2/(wl2 - C2*1e-3)) # delta of above eq. (sellmeier expansion)"""

'''
Function Def:
    Sellmeier-Herzberger dispersion Relation was used because it is a dielectric 
    material with vanishingly small extinction coefficient: 9.7525e^-9 
    Utilizing this coefficient, dispersion formula is shown via ref. 2)
    This calculation is for a non-gas filled PCF
Function Variables:
    Refractive Index: n
    Wave number: Δβ=β1+β2-2β0 e.g. we want Δβ=0=>group velocity dispersion read up on how deter rln
References:
    1) https://eng.libretexts.org/Bookshelves/Materials_Science/Supplemental_Modules_(Materials_Science)/Optical_Properties/Dispersion_Relation
    2) https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
    3)https://link.springer.com/article/10.1007/s00542-020-05019-w
Function def: Beta
    returns phase missmatch beta for each input frequency
    note this is not delta beta
Function variables:
    Vaccuum wave vector: k
    refractive index of fill gas: refrac_gas
    Effective/modal refractive index: nnm
    𝑢𝑚𝑛  is the nth zero of the mth-order Bessel function of the first kind, where 𝑚=𝑛=1 corresponds to the HE11 mode of the fiber.
    Several core radii, a, can be defined for the circle used to approximate the kagomé fiber core in Eq. (1),
References:
    Equations: Ultrafast nonlinear optics in gas-filled hollow-core photonic crystal fibers [Invited] (optica.org)
    check the ref for change in beta
'''
#Hollow PCF Veda's code

def beta1(w,radius,p,gas,T):#axial wavevector for phasematching calculation
    unm=2.405 # first zero of bessell function ( fundamental fiber mode approximation)
    k=w/c
    wl=2*pi*c/w
    Aeff=pi*radius**2 
    #will be an array of 3 values
    nmn=1+d_gas(w, gas, p, T)*rN_gas(p, T)-((pow(unm,2))/(2*pow(k,2)*pow(Aeff,2)))
    beta=(nmn*2*pi)/wl
    print('beta1:', beta)
    return beta
    #inverse meters

def beta_1_level(w,radius,p,gas,T):
    unm=2.405 # first zero of bessell function ( fundamental fiber mode approximation)
    Aeff=pi*radius**2 
    beta_1 = (1+d_gas(w, gas, p, T)*rN_gas(p, T)+((pow(unm,2)*pow(c,2))/(2*pow(w,2)*pow(Aeff,2))))/c
    print('beta_1_level:', beta_1)
    return beta_1

def beta_2_level(w,radius,p,gas,T):
    unm=2.405 # first zero of bessell function ( fundamental fiber mode approximation)
    Aeff=pi*radius**2 
    beta_2 = -(pow(unm,2)*c)/(pow(w,3)*pow(Aeff,2))
    print('beta_2_level:', beta_2)
    return beta_2

def beta_3_level(w,radius,p,gas,T):
    unm=2.405 # first zero of bessell function ( fundamental fiber mode approximation)
    Aeff=pi*radius**2 
    beta_3 = 3*(pow(unm,2)*c)/(pow(w,4)*pow(Aeff,2))
    print('beta_3_level:', beta_3)
    return beta_3

'''
NONLINEARITY
'''
'''
Function def:
    returns non-linear parameter
Function variables:
    Radius of beam: r
    Effective Area of laser beam: Aeff=pi*r^2
    Refractive index: n
    non-linear index: n2
    non-linear parameter: gamma
    pressure: p #atm
'''
#nonlinearity function
def chi3_gas(gas, p, T):
    """ Chi3 of a 'ideal gas' [m^2/V^2], mostly from Lehmeier 1985
        gas: gas [string]
        pressure: P [bar]
        temperature: T [K]
    """
    Chi3He = 3.43e-28 # at standard conditions
    Chi3 = {
        'He' : Chi3He,
        'Ne' : 1.8*Chi3He,
        'Ar' : 23.5*Chi3He,
        'Kr' : 64.0*Chi3He,
        'Xe' : 188.2*Chi3He,
            }
    return rN_gas(p, T)*4*Chi3[gas]

'''
Function Def:
    The non-linear parameter is dependant on chi-3, n2 and wavelength.  This relationship is described in detail in equations 
    2.3.28 and 1.3.3 in ref. 1).  The value of n2 was taken to be a constant. Note n2 varies over a range of wavelengths,
    this variation was so small it was neglected. The wavelength range chosen for the n2 value was between 486.1 and 656.3 nm.
Function variables:
    frequency: w
    Radius of beam: r
    Effective Area of laser beam: Aeff=pi*r^2
    non-linear index: n2
    non-linear parameter: γ
References:
Equations: Nonlinear Fiber Optics-Govind PP. Agrawal
n2 value: https://www.researchgate.net/publication/224846378_Measurement_of_nonlinear_refractive_index_coefficient_using_emission_spectrum_of_filament_induced_by_gigawatt-femtosecond_pulse_in_BK7_glass#:~:text=Measurement%20of%20nonlinear%20refractive%20index%20coefficient%20using%20emission,filament%20induced%20by%20gigawatt-femtosecond%20pulse%20in%20BK7%20glass
'''

#Hollow PCF Veda's code
def gamma2(w,radius,p,gas,T): #units [1/W/m]
    Aeff=pi*radius**2 #note the change to existing code
    n=np.sqrt(1+d_gas(w,gas=gas,p=p,T=T))
    n2=(3.0/4.0)*(chi3_gas(gas,p,T)/(eps0*c*pow(n,2)))
    gamma=n2*w/(c*Aeff)
    return gamma #true gamma factor is gamma_0*w, that I cut for simplicity



'''
PARAMETER CALCULATION
'''

"""
Function Def:
    function to evaluate the initial set of parameters of the ODE
    returns values of beta and gamma as array [[g,g,g],db]
"""
def get_params(omegas, radius, pressure, gas, T):
    wp, ws = omegas #wp=pump w, ws=signal w
    wa=2*wp-ws #wa=idler w
    db1=(beta1(ws, radius, pressure, gas=gas, T=T)+beta1(wa, radius, pressure, gas=gas, T=T)-2*beta1(wp, radius, pressure, gas=gas, T=T))*pow(10,-9) 
    print('db1:',db1)
    eta2=np.array([])
    for i in [wp,ws,wa]:
        eta2=np.append(eta2,gamma2(i,radius,pressure,gas=gas,T=T))
    print('gamma2:',eta2)
    beta_1p = beta_1_level(wp,radius,pressure, gas=gas, T=T)
    beta_1s = beta_1_level(ws,radius,pressure, gas=gas, T=T)
    beta_1i = beta_1_level(wa,radius,pressure, gas=gas, T=T)
    
    beta_2p = beta_2_level(wp,radius,pressure, gas=gas, T=T)
    beta_3p = beta_2_level(wp,radius,pressure, gas=gas, T=T)
    beta_2s = beta_2_level(ws,radius,pressure, gas=gas, T=T)
    beta_3s = beta_2_level(ws,radius,pressure, gas=gas, T=T)
    beta_2i = beta_2_level(wa,radius,pressure, gas=gas, T=T)
    beta_3i = beta_2_level(wa,radius,pressure, gas=gas, T=T)
    '''
    allows functioning of original and new code using same program
    '''
    #original code - Ravi - HCF
    #params = [eta0,db]

    #new code - Veda - Hollow - PCF
    params=[eta2, db1, beta_2p, beta_3p, beta_2s, beta_3s, beta_2i, beta_3i, beta_1p, beta_1s, beta_1i]

    return params


'''
ODE
fully frequency dependent ODE (no loss)
'''
'''
Function Def:
    Uses ODE for Continuos wave FWM from Cappellini et al. JOSA B  pp. 824-838 (1991)
    fully frequency dependent ODE (no loss)
    returns the electric field at input position z for one step

Function Variables:
    Initial Position: z
    Wave number: Δβ=β1+β2-2β0 e.g. we want Δβ=0
    Refractive index: n
    Wave length: lamda
    Non-linearity term: γ=2*pi*n2/(A0*lamda)
    Electric field: A
    Power/Intensity: P=A^2
'''
def rhs_full(z,A,params):
    eta = params[0]
    db = params[1]
    P=np.array([abs(i)**2 for i in A])
    dA=1j*eta[0]*((P[0] + 2*(P[1] + P[2]))*A[0] + 2*np.conj(A[0])*A[1]*A[2]*np.exp(+1j*db*z)) 
    dB=1j*eta[1]*((P[1] + 2*(P[0] + P[2]))*A[1] + np.conj(A[2])*A[0]*A[0]*np.exp(-1j*db*z)) 
    dC=1j*eta[2]*((P[2] + 2*(P[0] + P[1]))*A[2] + np.conj(A[1])*A[0]*A[0]*np.exp(-1j*db*z)) 
    return np.array([dA, dB, dC])
#----------------------------------------
def rhs_full_0(z,A,params):
    eta = params[0]
    db = params[1]
    P = []
    for i in range(0,len(A)):
        Power = abs(A[i])**2
        P.append(Power)


    # P=np.array([abs(i)**2 for i in A])
    dA=1j*eta[0]*((P[0] + 2*(P[1] + P[2]))*A[0] + 2*np.conj(A[0])*A[1]*A[2]*np.exp(+1j*db*z)) 
    dB=1j*eta[1]*((P[1] + 2*(P[0] + P[2]))*A[1] + np.conj(A[2])*A[0]*A[0]*np.exp(-1j*db*z)) 
    dC=1j*eta[2]*((P[2] + 2*(P[0] + P[1]))*A[2] + np.conj(A[1])*A[0]*A[0]*np.exp(-1j*db*z)) 
    return np.array([dA, dB, dC])

def rhs_full_1(A, un_1):
    a = []
    a = un_1
    return a

def rhs_full_2(un_1, un_2):
    b = []
    b = un_2
    return b

def rhs_full_3(z, A, un_1, un_2, params):
    dA, dB, dC = rhs_full_0(z, A, params)
    beta_2p = params[2]
    beta_3p = params[3]
    beta_2s = params[4]
    beta_3s = params[5]
    beta_2i = params[6]
    beta_3i = params[7]
    beta_1p = params[8]
    beta_1s = params[9]
    beta_1i = params[10]
    d_sp = beta_1s - beta_1p
    d_ip = beta_1i - beta_1p
    dA_2 = (dA-un_1[0]-(1j/2)*(beta_2p/pow(beta_1p,2))*un_2[0])/(1/6*beta_3p/pow(beta_1p, 3))
    dB_2 = (dB-(1-d_sp/beta_1p)*un_1[1]-(1j/2)*(beta_2s/pow(beta_1p,2))*un_2[1])/(1/6*beta_3s/pow(beta_1p, 3))
    dC_2 = (dC-(1-d_ip/beta_1p)*un_1[2]-(1j/2)*(beta_2i/pow(beta_1p,2))*un_2[2])/(1/6*beta_3i/pow(beta_1p, 3))

    return np.array([dA_2, dB_2, dC_2])

def RK4(z, A, un_1, un_2, dz, params):

    # zarray, Aarray, un_1_array, un_2_array = [], [], [], []
    
    # zarray.append(z)
    # Aarray.append(A)
    # un_1_array.append(un_1)
    # un_2_array.append(un_2)
        
    K_1 = dz * rhs_full_1(A, un_1)
    L_1 = dz * rhs_full_2(un_1, un_2)
    M_1 = dz * rhs_full_3(z, A, un_1, un_2, params)

    K_2 = dz * rhs_full_1(A, un_1+L_1/2)
    L_2 = dz * rhs_full_2(un_1, un_2+M_1/2)
    M_2 = dz * rhs_full_3(z+dz, A+K_1/2, un_1+L_1/2, un_2+M_1/2, params)

    K_3 = dz * rhs_full_1(A, un_1+L_2/2)
    L_3 = dz * rhs_full_2(un_1, un_2+M_2/2)
    M_3 = dz * rhs_full_3(z+dz, A+K_2/2, un_1+L_2/2, un_2+M_2/2, params)

    K_4 = dz * rhs_full_1(A, un_1+L_3)
    L_4 = dz * rhs_full_2(un_1, un_2+M_3)
    M_4 = dz * rhs_full_3(z+dz, A+K_3, un_1+L_3, un_2+M_3, params)
    


    A = A + 1/6 * (K_1 + 2 * K_2 + 2 *K_3 + K_4)
    un_1 = un_1 + 1/6 * (L_1 + 2 * L_2 + 2 * L_3 + L_4)
    un_2 = un_2 + 1/6 * (M_1 + 2 * M_2 + 2 * M_3 + M_4)
    
    return A

def prop_full2(L, p0, P0, delta=0.0001, z = 0.0):
    #g,db = p0
    Pmax=max(abs(P0))
    # print("params in prop_full: " +str(p0))
    gmax=max(p0[0])
    y=np.array(P0)
    print('y before sqrt:', y)
    y=np.sqrt(P0)   #e field
    print('y:',y)
    #this is the fix
    if gmax!=0:
        dz = 1e-3*delta/Pmax/gmax
    else:
        # dz=1e-5
        dz=1e-3
    dzmin=1e-5
    dzmin=1e-3
    dz=min(dz,dzmin)
    rho=0.0
    out_z=[]
    out_f=[]
    out_err=[]
    while z<L:
        out_z.append(z)
        out_f.append(y)
        out_err.append(rho)
        rho = 0.0
        while np.float32(rho)<1:
            dt = 2.0*dz
            y1 = RK4(z, y, np.zeros(3), np.zeros(3), dz, p0)#half step RK4(z, A, un_1, un_2, dz, params):
            y1 = RK4(z+dz, y, np.zeros(3), np.zeros(3), dz, p0)#half step consecutive
            y2 = RK4(z, y, np.zeros(3), np.zeros(3), 2*dz, p0)#double step
            # #print(y1)
            #print(y2)
            err = np.sqrt(np.sum(abs(y1-y2)**2))/30 + eps # error function
            rho = dz*delta/err
            dz = min(dz*rho**(0.25),2*dz)
            # dz = max(dzmin,dz)
        y = y1
        z += dz
    return np.array(out_z), np.array(out_f)


'''
Function Def:
    standard Runge-kutta 4th order step dy/dz=rhs(y,z)
    #Step size is constant and small value to plot accurately
    returns the next Electric field as a matrix for all 3 waves for some step size(dz)

References to understand RK4:
https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

Function Variables:
    position in fiber: z
    length of steps to get z: dz
    Refractive index: n
    Wave length: lamda
    Non-linearity parameter: γ
    ELectric Field: A
    Power distribution @ z for all  input waves: y=np.array([1.0, 1.0, 0.0]) 
    Wave number: Δβ=β1+β2-2β0 e.g. we want Δβ=0
    params=[γ, Δβ]
'''
def RK4full(y,z,dz,params):# params = pin 
    #standard Runge-kutta 4th order step dy/dz=rhs(y,z)
    k1 = rhs_full(z,y,params)
    k2 = rhs_full(z+0.5*dz,y+0.5*dz*k1,params)
    k3 = rhs_full(z+0.5*dz,y+0.5*dz*k2,params)
    k4 = rhs_full(z,y+dz*k3,params)
    return y+dz/6.0*(k1+2.0*k2+2.0*k3+k4)

'''
Function Def:
    returns position, electric field, error
    
Function Variables:
    Wave number: Δβ=β1+β2-2β0 e.g. we want Δβ=0
    Refractive index: n
    Wave length: lamda
    Non-linearity parameter: γ
    Length of fiber: L
    Step size:dz
    Initial postion: z
    Error control: delta => DO NOT TOUCH
    Electric field: A
    Power/Intensity: P=A^2
    Minimum step size to reduce computational time: dzmin
    params=[γ, Δβ]
    Position in Fiber: out_z
    Electric field of 3 waves in fiber: out_f
    Error: out_err
    Params: [γ, Δβ]
    Power array: P0
'''

def prop_full(L, p0, P0, delta=0.0001, z = 0.0):
    #g,db = p0
    Pmax=max(abs(P0))
    print("params in prop_full: " +str(p0))
    gmax=max(p0[0])
    y=np.array(P0)
    print('y before sqrt:', y)
    y=np.sqrt(P0)
    print('y:',y)
    #this is the fix
    if gmax!=0:
        dz = 1e-3*delta/Pmax/gmax
    else:
        dz=1e-5
    dzmin=1e-5
    dz=min(dz,dzmin)
    rho=0.0
    out_z=[]
    out_f=[]
    out_err=[]
    while z<L:
        out_z.append(z)
        out_f.append(y)
        out_err.append(rho)
        rho = 0.0
        while rho<1:
            dt = 2.0*dz
            y1 = RK4full(y,z,dz,p0)#half step
            y1 = RK4full(y1,z+dz,dz,p0)#half step consecutive
            y2 = RK4full(y,z,2*dz,p0)#double step
            #print(y1)
            #print(y2)
            err = np.sqrt(np.sum(abs(y1-y2)**2))/30 + eps # error function
            rho = dz*delta/err
            dz = min(dz*rho**(0.25),2*dz)
            #dz = max(dzmin,dz)
        y = y1
        z += dt
    return np.array(out_z), np.array(out_f), np.array(out_err)

'''
Optimizations
'''
def optimize_pressure(wavelengths,pressure_range,radius, num_points, gas_type,T):
    print("radius : " + str(radius))
    press=np.linspace(pressure_range[0],pressure_range[1],num_points) 
    db=[]
    for ps in press:
        a=get_params(2*pi*c/np.asarray(wavelengths),radius,ps,gas_type,T)
        db.append(a[1])
    db=np.array(db)
    ip_pm=np.argmin(np.abs(db))
    #print('db optimize pressure:', db)
    pin=get_params(2*pi*c/np.asarray(wavelengths),radius,press[ip_pm], gas_type, T)
    return pin, press[ip_pm]
# def run_optimize(wavelengths ,pressure_range, pw,radius, num_points, gas_type,T):
#     print("radius : " + str(radius))
#     pin,_ = optimize_pressure(wavelengths,pressure_range,radius, num_points, gas_type,T)
#     powers = np.array(pw)
#     Ps=np.sqrt(3e8*powers) # sqrt(peak powers of the pump, signal and idler beams)
#     z, A, err =prop_full(0.1,pin,Ps,delta=0.1,z=0.0)########propagator#########
#     return z, A, err

# Draw Gaaussian profile    
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)/0.4
    # mean1, sigma1 = 0, 1
    # x1 = np.linspace(mean1 - 6*sigma1, mean1 + 6*sigma1, 100)
    # y1 = normal_distribution(x1, mean1, sigma1)
    # plt.figure()
    # plt.plot(x1, y1, 'r', label='m=0,sig=1')

def main():
    starttime = time.time()
    wavelengths = [512.5e-9,1025e-9]
    # mean1, sigma1 = , 1
    # x1 = np.linspace(mean1 - 6*sigma1, mean1 + 6*sigma1, 100)
    # y1 = normal_distribution(x1, mean1, sigma1)
    # plt.figure()
    # plt.plot(x1, y1, 'r', label='m=0,sig=1')
    pressure_range = [0,4]
    radius_range = [100e-6]
    radius_range= np.array(radius_range)
    num_points = 50
    num_pow=int(1e07)
    num_pow_str=str(num_pow)
    gas_type=['Ar']
    T=293.15
    length = 50
    # arbitrary_max_z_size = 5000000
    # arbitrary_max_z_size = 97871

    powers =num_pow*np.array([1.0,1.0,0.0])
    
    #gas_type=['Kr']#['He']#, 'Ne']#, 'Ar', 'Xe']


    # power_list = np.zeros((len(gas_type),arbitrary_max_z_size,4,radius_range.shape[0]), dtype=np.cdouble)
    #pydevd warning: Computing repr of power_list (ndarray) was slow (took 0.54s)
    i = 0
    j = 0
    array_plist=np.array([])

    for gas in gas_type:
        # print(gas_type)
        print("gas type:", gas_type[j])
        print("Gas : " + gas)
        for radius in radius_range:
            pin,pressure = optimize_pressure(wavelengths,pressure_range,radius, num_points, gas,T)
            print("optimal pressure : " + str(pressure) + " atm")
            print("optimal params : " + str(pin))
            print("...running....")
            # for num in range(len(x1)):
            #     powers =num_pow*np.array([y1[num],y1[num],0.0])
            # z, A, err =prop_full(length,pin,powers,delta=0.1,z=0.0)
            z, A =prop_full2(length,pin,powers,delta=0.1,z=0.0)
            print("z shape:", str(z.shape[0]))
            # power_list[j,0:z.shape[0],1:4,i] = A
            # power_list[j,0:z.shape[0],0,i] = z
            i = i + 1
            # array_plist=np.append(array_plist,power_list)
        j = j + 1
    print(array_plist)
    endtime = time.time()
    print ("running time:", (endtime - starttime))
    print('...Graphing...')
    fig, axs = plt.subplots(len(gas_type),1)
    for i in range(len(gas_type)):
        #index = np.unravel_index(np.argmax(max_output_power_list[i],axis=None), max_output_power_list[i].shape)
        # A = np.squeeze(power_list[i,:,1:4,0])
        # z = np.squeeze(power_list[i,:,0,0])
        # temp = np.where(z[1:-1]==0)
        stop_index = z.shape[0]-1
        print('stop_index:', stop_index)
        #print(abs(A[0:stop_index, :]) ** 2)
        #print('values z:',z[0:stop_index].real)
        axs.plot(z,abs(A)**2)
        axs.plot(z,abs(A[:,0])**2+abs(A[:,1])**2+abs(A[:,2])**2)
        axs.set_ylabel('Peak Powers [W]')
        axs.set_xlabel('Fiber position [m]')
        plt.title('Ar: Input powers: '+ num_pow_str + ' (W)')
        plt.legend(["pump","signal","idler", "total power"],loc='best')
    plt.show()

main()
