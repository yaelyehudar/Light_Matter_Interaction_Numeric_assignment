import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants as const
import matplotlib.pyplot as plt

hbar = const.hbar 
pi = np.pi 
# ---------------------------------- #
# Part 1 - Tow level system dynamics 
# ---------------------------------- #

# ----------------------------------------- #
# 1. a. Time Dependent Schrodinger Equation 
# ----------------------------------------- #

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
OmegaR0 = 5.0

initial_state = np.array([0.0, 1.0], dtype=complex)  # Initial state: Cg(0) = 0, Ce(0) = 1

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# -------------------------------------------- #
# Rabi dynamics for Delta =0 and Delta = const
# -------------------------------------------- #

delta_cases = [0.0, 5.0]

for delta in delta_cases:
    sol = solve_ivp(schrodinger_rabi, t_span, initial_state, args=(delta, OmegaR0), t_eval=t_eval, rtol=1e-9, atol=1e-11)
          
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
    filename = f"Rabi_oscillations_delta_{delta}_Omega_{OmegaR0}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    

# ---------------------------------- #
# Scan detuning
# ---------------------------------- #

delta_values = np.linspace(-5 * OmegaR0, 5 * OmegaR0, 2000)

Pe_matrix = np.zeros((len(delta_values), len(t_eval)))

for i, delta in enumerate(delta_values):
    sol = solve_ivp(schrodinger_rabi, t_span, initial_state, args=(delta, OmegaR0), t_eval=t_eval, rtol=1e-9, atol=1e-11)

    t = sol.t
    Ce = sol.y[1]
    Pe = np.abs(Ce)**2

    Pe_matrix[i, :] = Pe

# ---------------------------------- #
# Plot Pe matrix
# ---------------------------------- #

plt.figure(figsize=(10,6))

plt.imshow(Pe_matrix, aspect="auto", origin="lower", extent=[t_eval[0], t_eval[-1], delta_values[0], delta_values[-1]])
plt.colorbar(label=r"$|C_e|^2$")
plt.xlabel("Time")
plt.ylabel(r"Detuning $\Delta$")
plt.savefig('output Pe Matrix')
plt.show()

# ---------------------------------- #
# x-axis cross sections
# ---------------------------------- #

selected_deltas = [-5 * OmegaR0, -2 * OmegaR0, 0, 2 * OmegaR0, 5 * OmegaR0]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes = axes.flatten()

for ax, delta_target in zip(axes, selected_deltas):

    index = np.argmin(np.abs(delta_values - delta_target))

    ax.plot(t_eval, Pe_matrix[index, :])
    ax.set_title(rf"$\Delta={delta_values[index]:.1f}$")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$P_e$")

# if subplot empty
for ax in axes[len(selected_deltas):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("Pe_cross_sections_delta_x.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------- #
# x-axis cross sections
# ---------------------------------- #

selected_times = [0, 2, 4, 6, 8]

plt.figure(figsize=(10,5))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes = axes.flatten()

for ax, t_target in zip(axes, selected_times):

    index = np.argmin(np.abs(t_eval - t_target))

    ax.plot(delta_values, Pe_matrix[:, index])
    ax.set_title(rf"$t={t_eval[index]:.1f}$")
    ax.set_xlabel(r"Detuning $\Delta$")
    ax.set_ylabel(r"$|C_e|^2$")

# if subplot empty
for ax in axes[len(selected_deltas):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("Pe_cross_sections_times_y.png", dpi=300, bbox_inches="tight")
plt.show()



# ------------------------------------- #
# 1. b. Optical Bloch Sphere Parameters 
# ------------------------------------- #

# need to solve the equation and get to each solution the Ce(t) Cg(t)
selected_deltas_bloch = [0, -5 * OmegaR0, 5 * OmegaR0 , 0.5 * OmegaR0]

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

# draw bloch sphere
phi = np.linspace(0, 2*pi, 60)
theta = np.linspace(0, pi, 30)

phi, theta = np.meshgrid(phi, theta) # now they both matrix (30, 60) (theta_ij, phi_ij)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

ax.plot_wireframe(x, y, z, alpha=0.1)

# coordinate axes
ax.plot([-1, 1], [0, 0], [0, 0])
ax.plot([0, 0], [-1, 1], [0, 0])
ax.plot([0, 0], [0, 0], [-1, 1])

# plot several trajectories

for delta in selected_deltas_bloch:
    sol = solve_ivp(schrodinger_rabi, t_span, initial_state, args=(delta, OmegaR0), t_eval=t_eval, rtol=1e-9, atol=1e-11)

    Cg = sol.y[0]
    Ce = sol.y[1]

    U = (Cg.conjugate() * Ce) + (Cg * Ce.conjugate())
    V = 1j * ((Cg.conjugate() * Ce) - (Cg * Ce.conjugate()))
    W = np.abs(Ce)**2 - np.abs(Cg)**2

    # U,V should theoretically be real
    U = np.real(U)
    V = np.real(V)
    W = np.real(W)

    ax.plot(U, V, W, label=rf"$\Delta={delta:.1f}$")

ax.set_xlabel("U")
ax.set_ylabel("V")
ax.set_zlabel("W")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.set_box_aspect([1, 1, 1])

ax.set_title("Trajectories on the Optical Bloch Sphere")
ax.legend()
plt.savefig("Bloch_sphere_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()




