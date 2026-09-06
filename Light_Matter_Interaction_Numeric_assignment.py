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
def schrodinger(t, state, Delta0, OmegaR0):
    Cg, Ce = state

    dCg_dt= -(1j/2) *((-Delta0 * Cg) + (OmegaR0 * Ce))
    dCe_dt= -(1j/2) *((Delta0 * Ce) + (OmegaR0.conjugate() * Cg))

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
    sol = solve_ivp(schrodinger, t_span, initial_state, args=(delta, OmegaR0), t_eval=t_eval, rtol=1e-9, atol=1e-11)
          
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
    

# # ---------------------------------- #
# # Scan detuning
# # ---------------------------------- #

# delta_values = np.linspace(-5 * OmegaR0, 5 * OmegaR0, 2000)

# Pe_matrix = np.zeros((len(delta_values), len(t_eval)))

# for i, delta in enumerate(delta_values):
#     sol = solve_ivp(schrodinger, t_span, initial_state, args=(delta, OmegaR0), t_eval=t_eval, rtol=1e-9, atol=1e-11)

#     t = sol.t
#     Ce = sol.y[1]
#     Pe = np.abs(Ce)**2

#     Pe_matrix[i, :] = Pe

# # ---------------------------------- #
# # Plot Pe matrix
# # ---------------------------------- #

# plt.figure(figsize=(10,6))

# plt.imshow(Pe_matrix, aspect="auto", origin="lower", extent=[t_eval[0], t_eval[-1], delta_values[0], delta_values[-1]])
# plt.colorbar(label=r"$|C_e|^2$")
# plt.xlabel("Time")
# plt.ylabel(r"Detuning $\Delta$")
# plt.savefig('output Pe Matrix')
# plt.show()

# # ---------------------------------- #
# # x-axis cross sections
# # ---------------------------------- #

# selected_deltas = [-5 * OmegaR0, -2 * OmegaR0, 0, 2 * OmegaR0, 5 * OmegaR0]

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# axes = axes.flatten()

# for ax, delta_target in zip(axes, selected_deltas):

#     index = np.argmin(np.abs(delta_values - delta_target))

#     ax.plot(t_eval, Pe_matrix[index, :])
#     ax.set_title(rf"$\Delta={delta_values[index]:.1f}$")
#     ax.set_xlabel("Time")
#     ax.set_ylabel(r"$P_e$")

# # if subplot empty
# for ax in axes[len(selected_deltas):]:
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig("Pe_cross_sections_delta_x.png", dpi=300, bbox_inches="tight")
# plt.show()

# # ---------------------------------- #
# # y-axis cross sections
# # ---------------------------------- #

# selected_times = [0, 2, 4, 6, 8]

# plt.figure(figsize=(10,5))

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# axes = axes.flatten()

# for ax, t_target in zip(axes, selected_times):

#     index = np.argmin(np.abs(t_eval - t_target))

#     ax.plot(delta_values, Pe_matrix[:, index])
#     ax.set_title(rf"$t={t_eval[index]:.1f}$")
#     ax.set_xlabel(r"Detuning $\Delta$")
#     ax.set_ylabel(r"$|C_e|^2$")

# # if subplot empty
# for ax in axes[len(selected_deltas):]:
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig("Pe_cross_sections_times_y.png", dpi=300, bbox_inches="tight")
# plt.show()



# ------------------------------------- #
# 1. b. Optical Bloch Sphere  
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
ax.plot([-1, 1], [0, 0], [0, 0], color='black')
ax.plot([0, 0], [-1, 1], [0, 0], color='black')
ax.plot([0, 0], [0, 0], [-1, 1], color='black')

# plot several trajectories

for delta in selected_deltas_bloch:
    sol = solve_ivp(schrodinger, t_span, initial_state, args=(delta, OmegaR0), t_eval=t_eval, rtol=1e-9, atol=1e-11)

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

# Custom axes
ax.quiver(1.0-1.2, 0, 0, 1.2, 0, 0, arrow_length_ratio=0.08, color='black')
ax.quiver(0, 1.0-1.2, 0, 0, 1.2, 0, arrow_length_ratio=0.08, color='black')
ax.quiver(0, 0, 1.0-1.2, 0, 0, 1.2, arrow_length_ratio=0.08, color='black')

# Labels
ax.text(1.0 + 0.05, 0, 0, 'U', fontsize=12)
ax.text(0, 1.0 + 0.05, 0, 'V', fontsize=12)
ax.text(0, 0, 1.0 + 0.05, 'W', fontsize=12)

plt.savefig("Bloch_sphere_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------------------- #
# 1. c. Known solutons of two level system 
# ----------------------------------------- #


# ------------------------------------- #
# Rosen Zender
# ------------------------------------- #

Omega0 = 5.0
Delta0 = 2.5
T = 2 * pi 
B = 10.0

def sech(x):
    return 1 / np.cosh(x)

def OmegaRZ(t, T, Omega0):
    return Omega0 * sech(t / T)

def deltaRZ(t, B, Delta0):
    return Delta0

# Time range
t_spanRZ = (-5*T, 5*T)
t_evalRZ = np.linspace(-5*T, 5*T, 1000)

def schrodingerRZ(t, state, Delta0, Omega0, B, T):
    Cg, Ce = state

    Omega_t = OmegaRZ(t, T, Omega0)
    delta_t = deltaRZ(t, B, Delta0)

    dCg_dt= -(1j/2) *((-delta_t * Cg) + (Omega_t * Ce))
    dCe_dt= -(1j/2) *((delta_t * Ce) + (Omega_t.conjugate() * Cg))

    return [dCg_dt, dCe_dt]


sol = solve_ivp(schrodingerRZ, t_spanRZ, initial_state, args=(Delta0, Omega0, B, T), t_eval=t_evalRZ, rtol=1e-9, atol=1e-11)

Cg = sol.y[0]
Ce = sol.y[1]

U = (Cg.conjugate() * Ce) + (Cg * Ce.conjugate())
V = 1j * ((Cg.conjugate() * Ce) - (Cg * Ce.conjugate()))
W = np.abs(Ce)**2 - np.abs(Cg)**2

# U,V should theoretically be real
U = np.real(U)
V = np.real(V)
W = np.real(W)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(x, y, z, alpha=0.1)

# coordinate axes
ax.plot([-1, 1], [0, 0], [0, 0], color='black')
ax.plot([0, 0], [-1, 1], [0, 0], color='black')
ax.plot([0, 0], [0, 0], [-1, 1], color='black')


ax.plot(U, V, W, label=rf"$\Delta(t)= \Delta_0, \Omega(t) = \Omega_0 sech(t/T)$")

ax.set_xlabel("U")
ax.set_ylabel("V")
ax.set_zlabel("W")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.set_box_aspect([1, 1, 1])

ax.set_title("Trajectories on the Optical Bloch Sphere Rosen Zender Solution")
ax.legend()

# Custom axes
ax.quiver(1.0-1.2, 0, 0, 1.2, 0, 0, arrow_length_ratio=0.08, color='black')
ax.quiver(0, 1.0-1.2, 0, 0, 1.2, 0, arrow_length_ratio=0.08, color='black')
ax.quiver(0, 0, 1.0-1.2, 0, 0, 1.2, arrow_length_ratio=0.08, color='black')

# Labels
ax.text(1.0 + 0.05, 0, 0, 'U', fontsize=12)
ax.text(0, 1.0 + 0.05, 0, 'V', fontsize=12)
ax.text(0, 0, 1.0 + 0.05, 'W', fontsize=12)

plt.savefig("Bloch_sphere_trajectories Rosen Zender Solution.png", dpi=300, bbox_inches="tight")
plt.show()



# ------------------------------------- #
# Allen Eberly
# ------------------------------------- #

Omega0 = 5.0
Delta0 = 2.5
T = 2 * pi 
B = 10.0

def OmegaAE(t, T, Omega0):
    return Omega0 * sech(t / T)

def deltaAE(t, B, T):
    return B * np.tanh(t / T)

# Time range
t_spanAE = (-5*T, 5*T)
t_evalAE = np.linspace(-5*T, 5*T, 1000)

def schrodingerAE(t, state, Omega0, B, T):
    Cg, Ce = state

    Omega_t = OmegaAE(t, T, Omega0)
    delta_t = deltaAE(t, B, T)

    dCg_dt= -(1j/2) *((-delta_t * Cg) + (Omega_t * Ce))
    dCe_dt= -(1j/2) *((delta_t * Ce) + (Omega_t.conjugate() * Cg))

    return [dCg_dt, dCe_dt]


sol = solve_ivp(schrodingerAE, t_spanAE, initial_state, args=(Omega0, B, T), t_eval=t_evalAE, rtol=1e-9, atol=1e-11)

Cg = sol.y[0]
Ce = sol.y[1]

U = (Cg.conjugate() * Ce) + (Cg * Ce.conjugate())
V = 1j * ((Cg.conjugate() * Ce) - (Cg * Ce.conjugate()))
W = np.abs(Ce)**2 - np.abs(Cg)**2

# U,V should theoretically be real
U = np.real(U)
V = np.real(V)
W = np.real(W)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(x, y, z, alpha=0.1)

# coordinate axes
ax.plot([-1, 1], [0, 0], [0, 0], color='black')
ax.plot([0, 0], [-1, 1], [0, 0], color='black')
ax.plot([0, 0], [0, 0], [-1, 1], color='black')


ax.plot(U, V, W, label=rf"$\Delta(t)= Btanh(t/T), \Omega(t) = \Omega_0sech(t/T)$")

ax.set_xlabel("U")
ax.set_ylabel("V")
ax.set_zlabel("W")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.set_box_aspect([1, 1, 1])

ax.set_title("Trajectories on the Optical Bloch Sphere Allen Eberly Solution")
ax.legend()

# Custom axes
ax.quiver(1.0-1.2, 0, 0, 1.2, 0, 0, arrow_length_ratio=0.08, color='black')
ax.quiver(0, 1.0-1.2, 0, 0, 1.2, 0, arrow_length_ratio=0.08, color='black')
ax.quiver(0, 0, 1.0-1.2, 0, 0, 1.2, arrow_length_ratio=0.08, color='black')

# Labels
ax.text(1.0 + 0.05, 0, 0, 'U', fontsize=12)
ax.text(0, 1.0 + 0.05, 0, 'V', fontsize=12)
ax.text(0, 0, 1.0 + 0.05, 'W', fontsize=12)

plt.savefig("Bloch_sphere_trajectories Allen Eberly Solution.png", dpi=300, bbox_inches="tight")
plt.show()


