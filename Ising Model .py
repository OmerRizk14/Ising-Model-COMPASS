import numpy as np
import matplotlib.pyplot as plt
#import numba
#from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import tkinter as tk
from tkinter import ttk

N = 50

# E/j = - collection <i,j> Qi.Qj

def get_energy(lattice):
    kern = generate_binary_structure(2,1)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum()

#@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy
        E_i = 0
        E_f = 0
        if x>0:
            E_i += -spin_i*spin_arr[x-1,y]
            E_f += -spin_f*spin_arr[x-1,y]
        if x<N-1:
            E_i += -spin_i*spin_arr[x+1,y]
            E_f += -spin_f*spin_arr[x+1,y]
        if y>0:
            E_i += -spin_i*spin_arr[x,y-1]
            E_f += -spin_f*spin_arr[x,y-1]
        if y<N-1:
            E_i += -spin_i*spin_arr[x,y+1]
            E_f += -spin_f*spin_arr[x,y+1]
        
        # 3 / 4. change state with designated probabilities
        dE = E_f-E_i
        if (dE>0)*(np.random.random() < np.exp(-BJ*dE)):
            spin_arr[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_arr[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy


# GUI Button Trigger Function
def click_run_button():
    global Bj, atoms_positive_charge, steps, lattice_n
    
    # Reading numbers from Tkinter Entry fields
    Bj = float(entry_temp.get())
    atoms_positive_charge = float(entry_charge.get()) / 100
    steps = int(entry_steps.get())
    
    # Initial conditions using the values from GUI
    init_random = np.random.random((N,N))
    lattice_n = np.zeros((N,N))
    lattice_n[init_random >= atoms_positive_charge] = 1
    lattice_n[init_random < atoms_positive_charge] = -1
    
    # Running your functions
    spins, energies = metropolis(lattice_n, steps, Bj, get_energy(lattice_n))
    
    # Plotting results (Opens in a new separate window)
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    ax = axes[0]
    ax.plot(spins/N**2)
    ax.set_xlabel('Algorithm Time Steps')
    ax.set_ylabel(r'Average Spin $\bar{m}$')
    ax.grid()
    ax = axes[1]
    ax.plot(energies)
    ax.set_xlabel('Algorithm Time Steps')
    ax.set_ylabel(r'Energy $E/J$')
    ax.grid()
    fig.tight_layout()
    fig.suptitle(f'Evolution of Average Spin and Energy for $\\beta J=${Bj}', y=1.07, size=18)

    plt.show()

# Tkinter user interface window
root = tk.Tk()
root.title("Input Window")
root.geometry("350x250")

# 1. temprature value field
ttk.Label(root, text="Enter the temprature value (0.1 - 4.0):").pack(pady=5)
entry_temp = ttk.Entry(root)
entry_temp.insert(0, "0.2")
entry_temp.pack(pady=5)

# 2. Percentage field
ttk.Label(root, text="Enter the percentage of atoms with charge +1: %").pack(pady=5)
entry_charge = ttk.Entry(root)
entry_charge.insert(0, "55")
entry_charge.pack(pady=5)

# 3. Steps field
ttk.Label(root, text="Enter the number of steps:").pack(pady=5)
entry_steps = ttk.Entry(root)
entry_steps.insert(0, "100000")
entry_steps.pack(pady=5)

# Run Button
btn_run = ttk.Button(root, text="Run Simulation", command=click_run_button)
btn_run.pack(pady=15)

root.mainloop()
