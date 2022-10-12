import wave
import numpy as np
import sys
from scipy import pi
import matplotlib.pyplot as plt
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
Dispersion
'''
def rN_gas(p, T=293.15): # for moderate pressure/Temperature(e.g. <50bar), for higher pressure need better compressibility model (non-ideal gas)
    T0 = 273.15
    p0 = 1.01325
    return p/p0*T0/T

# dispersion function
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
    wl =2*pi*c/w # wavelegnth uncessary calculation just for clarity
    wl2 = (wl*1e6)**2 # usal spectroscopist units
    return rN*(B1*1e-8*wl2/(wl2 - C1*1e-6)+ B2*1e-8*wl2/(wl2 - C2*1e-3)) # delta of above eq. (sellmeier expansion)

def beta(w,radius,p,gas='He',T=293.15):#axial wavevector for phasematching calculation
    unm=2.405 # first zero of bessell function ( fundamental fiber mode approximation)
    return np.sqrt((w/c)**2*(1+d_gas(w,gas,p=p,T=T))-(unm/radius)**2)

'''
Nonlinearity
'''
#nonlinearity function
def chi3_gas(gas, p=1.0, T=293.15):
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
    
def gamma0(radius,p,gas='Ar',T=293.15): #effective nonlinear  parameter 
    n2=3.0*chi3_gas(gas, p=p, T=T)/(4.0*eps0*c) # nonlinear refractive index
    Aeff=1.4971689488*(radius**2) #effective area APPROXIMATION (~1.5 fiber radius squared)
    return n2/(c*Aeff) #true gamma factor is gamma_0*w/n**2, that I cut for simplicity here  

#true nonlinear coefficient g from nonlinear refractive index n2 (see Agrawal for example)
def gamma(w,radius,p,gas='He',T=293.15): #nonlinear  coefficient [1/W/m]
    Aeff=1.5**radius**2 #1.4971689488*radius**2 #effective area APPROXIMATION (~1.5 fiber radius squared)
    n0=np.sqrt(1+d_gas(w,gas=gas,p=p,T=T))
    n2=3.0/4.0*chi3_gas(gas=gas, p=p, T=T)/(eps0*c*n0**2) # nonlinear refractive index
    return n2*w/(c*Aeff) #true gamma factor is gamma_0*w, that I cut for simplicity

'''
Parameter Calculation
this guy calculate the gamma (eta) and phasemismatch (db) for the coupled FWM equations
'''
#function to evaluate the initial set of parameters of the ODE
def get_params(omegas, radius, pressure, gas='Ar', T=293.15):
    wp, ws = omegas 
    wa=2*wp-ws
    #print(wp)
    db=beta(ws, radius, pressure, gas=gas, T=T)+beta(wa, radius, pressure, gas=gas, T=T)-2*beta(wp, radius, pressure, gas=gas, T=T)
    eta0=np.array([ i*gamma0(radius, pressure, gas=gas, T=T ) for i in ([wp,ws,wa])]) #freq-dep gamma here instead of in gamma func commented above
    eta=np.array([ gamma(i,radius, pressure, gas=gas, T=T ) for i in ([wp,ws,wa])]) #freq-dep gamma here instead of in gamma func commented above
    params = [eta0,db]
    #params = [eta,db] #just quickly tried adding this in but didn't work (Jack)
    #so according to notes above this eta calculated from gamma should be the same as gamm0*w but not yielding same result

    return params


'''
ODE
fully frequency dependent ODE (no loss)
'''
def rhs_full(z,A,params):
    eta = params[0]
    db = params[1]
    P=np.array([abs(i)**2 for i in A])
    dA=1j*eta[0]*((P[0] + 2*(P[1] + P[2]))*A[0] + 2*np.conj(A[0])*A[1]*A[2]*np.exp(+1j*db*z)) 
    dB=1j*eta[1]*((P[1] + 2*(P[0] + P[2]))*A[1] + np.conj(A[2])*A[0]*A[0]*np.exp(-1j*db*z)) 
    dC=1j*eta[2]*((P[2] + 2*(P[0] + P[1]))*A[2] + np.conj(A[1])*A[0]*A[0]*np.exp(-1j*db*z)) 
    return np.array([dA, dB, dC])

    '''
    These equations do not consider the bandwidth of the Laser. Need to work on Efficient stratergies
    to incorporate the bandwidth into the code.
    Some important resources:
    1. Yongzhong Li et al 2006 J. Opt. A: Pure Appl. Opt. 8 689 (Has the Full equations and solutions)
    2. The py-fmas python package solves for the z-propagation dynamics of spectrally broad ultrashort
    optical pulses in single mode nonlinear waveguides in terms of a nonlinear unidirectional propagation
    model for the analytic signal of the optical field.
    Link: https://omelchert.github.io/py-fmas/
    3. A Github link for Self-Phase modulation to get some ideas
    Link: https://github.com/ry-dgel/self-phase
    '''

def RK4full(y,z,dz,params):
    #standard Runge-kutta 4th order step dy/dz=rhs(y,z)
    k1 = rhs_full(z,y,params)
    k2 = rhs_full(z+0.5*dz,y+0.5*dz*k1,params)
    k3 = rhs_full(z+0.5*dz,y+0.5*dz*k2,params)
    k4 = rhs_full(z,y+dz*k3,params)
    return y+dz/6.0*(k1+2.0*k2+2.0*k3+k4)

'''
Propagation
wrapped in a function to call for optimization
'''
def prop_full(L, p0, P0, delta=0.0001, z = 0.0):
    #g,db = p0
    Pmax=max(abs(P0))
    print("pressure : " +str(p0))
    gmax=max(p0[0])
    y=np.array(P0)
    dz = 1e-3*delta/Pmax/gmax
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
def optimize_pressure(wavelengths = [400e-9,800e-9],pressure_range = [0,6],radius = 100e-6, num_points = 50, gas_type='He',T=293.15):
    print("radius : " + str(radius))
    press=np.linspace(pressure_range[0],pressure_range[1],num_points) 
    db=[]
    for ps in press:
        a=get_params(2*pi*c/np.asarray(wavelengths),radius,ps,gas=gas_type,T=293.15)
        db.append(a[1])
    db=np.array(db)
    ip_pm=np.argmin(np.abs(db))
    pin=get_params(2*pi*c/np.asarray(wavelengths),radius,press[ip_pm],gas=gas_type,T=293.15)
    return pin, press[ip_pm]
def run_optimize(wavelengths = [400e-9,800e-9],pressure_range = [14,18], pw = [3e8, 3e8, 0.0],radius = 25e-6, num_points = 50, gas_type='He',T=293.15):
    print("radius : " + str(radius))
    pin,_ = optimize_pressure(wavelengths,pressure_range,radius, num_points, gas_type,T)
    powers = np.array(pw)
    Ps=np.sqrt(3e8*powers) # sqrt(peak powers of the pump, signal and idler beams)
    z, A, err =prop_full(0.1,pin,Ps,delta=0.1,z=0.0)########propagator#########
    return z, A, err

'''
Simplified phase equations
expecting list of wavelengths and either a phase_signal as a function of omega or infrared phase list for cubic phase
Akanshya's code should be made so that it can be integrated right-in for a rough calculation of the Phase transfer

def simplified_phase(wavelengths=[400e-9, 800e-9], phi_signal=None, infrared_phase_list=None):
    omega_signal = 2*np.pi*c/wavelengths[1]
    omega_pump = 2*np.pi*c/wavelengths[0]
    omega_shg = 2*omega_signal
    omega_fwm = 2*omega_shg-omega_pump

    if (phi_signal==None): 
        phi_signal = infrared_phase_list[0]*omega**3+infrared_phase_list[1]*omega**2+infrared_phase_list[2]*omega+infrared_phase_list[3]
    p1 = (omega-omega_fwm)*phi_signal
    p2 = 0.5*((omega-omega_fwm)**2)*phi_signal**2
    p3 = ((omega-omega_fwm)**2)*phi_signal**2
'''
#def differential_substitution(signal, differential_order = 1, omega_sub):
#    signal_dif = np.diff(signal, differential_order)
    

'''
Testing
'''
wavelengths = [512.5e-9,1025e-9]
pressure_range = [0,4]
radius_range = [100e-6]
radius_range= np.array(radius_range)
num_points = 50
powers = np.sqrt(4e08*np.array([1.0,1.0,0.0]))
gas_type=['Kr']#['He']#, 'Ne']#, 'Ar', 'Kr', 'Xe']
T=293.15
length = 2
arbitrary_max_z_size = 10000
power_list = np.zeros((len(gas_type),arbitrary_max_z_size,4,radius_range.shape[0]), dtype=np.cdouble)

i = 0
j = 0
for gas in gas_type:
    print("Gas : " + gas)
    for radius in radius_range:
        pin,pressure = optimize_pressure(wavelengths,pressure_range,radius, num_points, gas,T)
        print("pressure : " + str(pressure))

        print("...running....")
        z, A, err =prop_full(length,pin,powers,delta=0.1,z=0.0)
        print("z shape" + str(z.shape[0]))
        power_list[j,0:z.shape[0],1:4,i] = A
        power_list[j,0:z.shape[0],0,i] = z
        i = i + 1
    j = j + 1
    ########propagator#########
    
    

'''
#Select optimized length and radius

#1 loop through each element in list and select optimized power and track index
#2 loop over those maximums to find optimized radius

max_output_power_list = np.zeros((len(gas_type),radius_range.shape[0]), dtype=np.float64)
max_power_length_list = np.zeros((len(gas_type),radius_range.shape[0]), dtype=np.float64)
#length_list = np.linsapce(0,1, num=50, endpoint=True)

#for j in range(len(gas_type)):
for i in range(0,power_list.shape[2]):
    max_output_power_list[:,i]=np.max(abs(power_list[:,:,3,i])**2,axis=0)
    max_power_length_list[:,i]=power_list[:,:,0,i][np.unravel_index(np.argmax(abs(power_list[:,:,3,i])**2,axis=0),power_list[:,:,3,i].shape )]

final_max_power = np.max(max_output_power_list,axis=0)

#radius_index = np.unravel_index(np.argmax(max_output_power_list[j],axis=0), max_output_power_list.shape)
print("Radius index constant right now\n")
radius_index = 0
final_length = max_power_length_list[:,radius_index]

print("Max output power is "+str(final_max_power)+ "at length " + str(final_length)+"\n")
print("Radius is " + str(radius_range[radius_index]))

np.save("max_output_powers", max_output_power_list,allow_pickle=True)
np.save("max_power_lengths", max_power_length_list,allow_pickle=True)


'''
fig, axs = plt.subplots(len(gas_type),1)
for i in range(len(gas_type)):
    #index = np.unravel_index(np.argmax(max_output_power_list[i],axis=None), max_output_power_list[i].shape)
    A = np.squeeze(power_list[i,:,1:4,0])
    z = np.squeeze(power_list[i,:,0,0])
    temp = np.where(z[1:-1]==0)
    stop_index = temp[0][0]
    print(stop_index)
    #print(abs(A[0:stop_index, :]) ** 2)
    print(z[0:stop_index].real)
    axs.plot(z[0:stop_index],abs(A[0:stop_index,:])**2)
    axs.plot(z[0:stop_index],abs(A[0:stop_index,0])**2+abs(A[0:stop_index,1])**2+abs(A[0:stop_index,2])**2)
    axs.set_ylabel('Peak Powers [W]')
    axs.set_xlabel('Fiber position [m]')
plt.show()

'''
plt.figure()
plt.plot(z,abs(A)**2)
plt.plot(z,abs(A[:,0])**2+abs(A[:,1])**2+abs(A[:,2])**2)
plt.ylabel('Peak Powers [W]')
plt.xlabel('Fiber position [m]')
plt.show()
'''