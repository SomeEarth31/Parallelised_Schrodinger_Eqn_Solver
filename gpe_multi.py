import numpy as np
import copy
from tqdm import tqdm
import time
import threading

#================================================================================
#                       Change the following parameters
#================================================================================

# Set grid size: a) 1D: set Ny,Nz = 1 b) 2D: set Nz = 1 
N = 512
Nx = N
Ny = N #256
Nz = N

# Set box length
L = 16
Lx = L
Ly = L
Lz = L

# Set maximum time and dt
tmax = 2                
dt = 0.5


# Choose the value of the non linerarity
g = 0

nstep = int(tmax/dt)

#================================================================================
# Set initial condition and potential

def set_initialcond():
    if dimension==1:
        '''
        Initial wavefunction
        '''
        wfc = ((1/np.pi)**(1/4))*np.exp(-(x**2)/2)+0j    #1D initial condition   
            
        '''
        potential
        '''
        pot = 1/2*(x**2)
        
    if dimension==2:
        '''
        Initial wavefunction
        '''
        wfc = ((1/np.pi)**(1/2))*np.exp(-(y_mesh**2 + x_mesh**2)/2)+0j    #2D initial condition   
            
        '''
        potential
        '''
        pot = 1/2*(x_mesh**2+y_mesh**2)



    elif dimension==3:
        '''
        Initial wavefunction
        '''
        wfc = ((1/np.pi)**(3/4))*np.exp(-(z_mesh**2+y_mesh**2+x_mesh**2)/2)+0j  # 3D initial condition

        '''
        potential
        '''
        pot = 1/2*(x_mesh**2+y_mesh**2+z_mesh**2)
    return wfc, pot



#================================================================================

if Nx != 1 and Ny == 1 and Nz == 1:
    dimension = 1

elif Nx != 1 and Ny != 1 and Nz == 1:
    dimension = 2

elif Nx != 1 and Ny != 1 and Nz != 1:
    dimension = 3

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz


x=np.arange(-Nx//2+1, Nx//2+1) * dx
volume = dx

if dimension == 2:
    y=np.arange(-Ny//2+1, Ny//2+1) * dy
    volume = dx * dy
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    
elif dimension == 3:
    y=np.arange(-Ny//2+1, Ny//2+1) * dy
    z = np.arange(-Nz//2+1, Nz//2+1) * dz
    volume = dx * dy * dz
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')


def laplacian(psi):
    temp=copy.deepcopy(psi) 
    if dimension == 1:
        # psi[0], psi[-1] = 0, 0
        temp = (np.roll(psi,-1)-2*psi+np.roll(psi,1))/dx**2
        # temp[1:-1]= (psi[2:]-2*psi[1:-1]+psi[:-2])/dx**2
    
    elif dimension == 2:
        # temp = (np.roll(psi,-1,axis=0)-2*psi+np.roll(psi,1,axis=0))/dx**2
        # temp += (np.roll(psi,-1,axis=1)-2*psi+np.roll(psi,1,axis=1))/dy**2
        temp[0,:], temp[-1,:] = 0, 0
        temp[:,0], temp[:,-1] = 0, 0
        temp[1:-1,:]= (psi[2:,:]-2*psi[1:-1,:]+psi[:-2,:])/dx**2
        temp[:,1:-1]+=(psi[:,2:]-2*psi[:,1:-1]+psi[:,:-2])/dy**2
    
    elif dimension == 3:
        # temp = (np.roll(psi,-1,axis=0)-2*psi+np.roll(psi,1,axis=0))/dx**2
        # temp += (np.roll(psi,-1,axis=1)-2*psi+np.roll(psi,1,axis=1))/dy**2
        # temp += (np.roll(psi,-1,axis=2)-2*psi+np.roll(psi,1,axis=2))/dz**2
        temp[1:-1,:,:]= (psi[2:,:,:]-2*psi[1:-1,:,:]+psi[:-2,:,:])/dx**2
        temp[:,1:-1,:]+=(psi[:,2:,:]-2*psi[:,1:-1,:]+psi[:,:-2,:])/dy**2
        temp[:,:,1:-1]+=(psi[:,:,2:]-2*psi[:,:,1:-1]+psi[:,:,:-2])/dz**2
    
    return temp




def integral(psi):
    return volume * np.sum(psi)

def norm(psi):
    return integral(np.abs(psi)**2)

def energy(psi, pot):
    z=np.conjugate(psi)*(-1/2*laplacian(psi)+(pot+g*np.abs(psi)**2/2)*psi)
    return integral(z.real)




#================================================================================
# Numerical schemes for single step time evolution
#================================================================================

def compute_RHS(psi, pot):
    return -1j*(-1/2*laplacian(psi)+(pot + g * np.abs(psi)**2)*psi)


# RK2 schemes for single step time evolution
def sstep_RK2(psi, pot):
    k1 = psi + compute_RHS(psi, pot) * dt/2
    return psi + compute_RHS(k1, pot) * dt


def sstep_RK4(psi, pot):
    k1 = compute_RHS(psi, pot)
    k2 = compute_RHS(psi + k1 * dt/2, pot)
    k3 = compute_RHS(psi + k2 * dt/2, pot)
    k4 = compute_RHS(psi + k3 * dt, pot)
    return psi + (k1 + 2*k2 + 2*k3 + k4) * dt/6   


def parallel_sstep_RK4(psi, pot, num_threads):
    step_size = int(len(psi) / num_threads)
    threads = []

    for i in range(num_threads):
        start_idx = i * step_size
        end_idx = start_idx + step_size if i < num_threads - 1 else len(psi)
        thread = threading.Thread(target=sstep_RK4, args=(psi[start_idx:end_idx], pot[start_idx:end_idx]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return psi

#=================================================================================
wfc, pot = set_initialcond()
print('norm:', norm(wfc))
print('energy:', energy(wfc, pot))
j=0

t_temp = dt
start = time.time()
for i in tqdm(range(nstep)):
    
    wfc = parallel_sstep_RK4(wfc, pot, 4)
    # print('t: ', t_temp)
    
    # if (tstep[j]-t_temp)/dt<dt:
    if i % 100 == 0:
        print('\n----------------------')
        print('t: ', t_temp)
        print('norm: ', norm(wfc))
        print('energy: ', energy(wfc, pot))
        j=j+1
    t_temp =t_temp + dt

end = time.time()

print('\n----------------------')
print("\nThe time of execution of above 1d", i, " program is :",
    (end-start) * 10**3, "ms\n")
print('----------------------')
