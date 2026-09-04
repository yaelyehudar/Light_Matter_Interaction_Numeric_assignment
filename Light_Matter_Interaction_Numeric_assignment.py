import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants as const
import matplotlib.pyplot as plt

hbar = const.hbar 

# ---------------------------------- #
# The Dependent Schrodinger Equation #
# ---------------------------------- #

# the states that does oscilations are Cg(t) and Ce(t)

# ---------------------------------- #
# Define the ODEs 
# ---------------------------------- #

def schrodinger_rabi(t, state, delta, OmegaR0):
    Cg, Ce = state

    # can write this as matrix?????
    dCg_dt= 1j * hbar/2 *((-delta * Cg) + (OmegaR0 * Ce))
    dCe_dt= 1j * hbar/2 *((delta * Ce) + (OmegaR0.conjugate() * Cg))

    return [dCg_dt, dCe_dt]

# ---------------------------------- #
# Parameters and initial states
# ---------------------------------- #
#OmegaR0 = [5.0, 10.0]
OmegaR0 = [5.0]

#delta = [0.0, 5.0, 10.0]
delta = np.linspace(-5*OmegaR0[0], 5*OmegaR0[0], 2) # maybe to go over all omega R0? if not 5 dots

#OmegaR = np.sqrt(OmegaR0[0]**2 + delta[0]**2)

initial_state = [1.0 + 0.0j, 0.0 + 0.0j]  # Initial state: Cg(0) = 1, Ce(0) = 0

t_span = (0, 100)

# ---------------------------------- #
# solutions 
# ---------------------------------- #
solutions = []
for i in range(len(delta)):
    for k in range(len(OmegaR0)):
        sol_ik = solve_ivp(schrodinger_rabi, t_span, initial_state, args=(delta[i], OmegaR0[k]), dense_output=True)
        solutions.append(sol_ik)
        t_ik = sol_ik.t
        Cg_t_ik = sol_ik.y[0]
        Ce_t_ik = sol_ik.y[1]
        norm_ik = np.sqrt(np.abs(Cg_t_ik)**2 + np.abs(Ce_t_ik)**2)
        comp = np.all(norm_ik == norm_ik[0])

        print("----------------------------")
        print("Solution ", i, k, " with delta: ", delta[i], " OmegaR0: ", OmegaR0[k])
        print("Cg(t): ", Cg_t_ik)
        print("Ce(t): ", Ce_t_ik)
        print("Normalization check (should be close to 1): ", np.average(norm_ik), ". Are all the values equal? ", comp)
        print()

        plt.figure(figsize = (10,4))
        plt.plot(t_ik, Cg_t_ik, linewidth=0.8)
        plt.plot(t_ik, Ce_t_ik, linewidth=0.8)
        plt.title('Population vs. Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Population')
        plt.legend()
        plt.show()

# Plot???????????


# ---------------------------------- #
# Plot
# ---------------------------------- #


