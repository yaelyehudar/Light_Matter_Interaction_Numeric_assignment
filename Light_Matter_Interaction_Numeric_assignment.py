import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants as const
import matplotlib.pyplot as plt

hbar = const.hbar 

# ---------------------------------- #
# Time Dependent Schrodinger Equation #
# ---------------------------------- #

# the states that does oscilations are Cg(t) and Ce(t)

# ---------------------------------- #
# Define the ODEs 
# ---------------------------------- #
def schrodinger_rabi(t, state, delta, OmegaR0):
    Cg, Ce = state

    # can write this as matrix?????
    dCg_dt= -(1j/2) *((-delta * Cg) + (OmegaR0 * Ce))
    dCe_dt= -(1j/2) *((delta * Ce) + (OmegaR0.conjugate() * Cg))

    return [dCg_dt, dCe_dt]

# ---------------------------------- #
# Parameters and initial states
# ---------------------------------- #
OmegaR0 = [5.0]

initial_state = np.array([0.0, 1.0], dtype=complex)  # Initial state: Cg(0) = 0, Ce(0) = 1

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# -------------------------------------------- #
# Rabi dynamics for Delta =0 and Delta = const
# -------------------------------------------- #

delta_cases = [0.0, 5.0]

for delta in delta_cases:
    sol = solve_ivp(schrodinger_rabi, t_span, initial_state, args=(delta, OmegaR0[0]), t_eval=t_eval, rtol=1e-9, atol=1e-11)
          
    t = sol.t
    Cg = sol.y[0]
    Ce = sol.y[1]

    Pg = np.abs(Cg)**2
    Pe = np.abs(Ce)**2

    norm = Pg + Pe
    
    print("----------------------------")
    print(f"Delta = {delta}")
    print("Average normalization:", np.mean(norm))
    print("Maximum normalization error:", np.max(np.abs(norm - 1)))
    
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, Pg, label=r"$|C_g|^2$")
    plt.plot(sol.t, Pe, label=r"$|C_e|^2$")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title(rf"Rabi oscillations, $\Delta={delta}$")
    plt.legend()
    plt.show()
    

# ---------------------------------- #
# Scan detuning
# ---------------------------------- #


