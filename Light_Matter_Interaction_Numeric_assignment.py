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
# Define the ODEs manually
# ---------------------------------- #
def schrodinger_manual_2_level(Delta0, OmegaR0, t_arr, Cg0, Ce0):

    Cg_m = np.zeros(len(t_arr), dtype=complex)
    Ce_m = np.zeros(len(t_arr), dtype=complex)

    Cg_m[0] = Cg0
    Ce_m[0] = Ce0

    for i in range(len(t_arr)-1):
        dt = t_arr[i+1] - t_arr[i]

        Cg_m[i+1] = dt * (-(1j/2) *((-Delta0 * Cg_m[i]) + (OmegaR0 * Ce_m[i]))) + Cg_m[i]
        Ce_m[i+1] = dt * (-(1j/2) *((Delta0 * Ce_m[i]) + (OmegaR0.conjugate() * Cg_m[i]))) + Ce_m[i]

    return Cg_m, Ce_m

# ---------------------------------- #
# Define the ODEs using numpy
# ---------------------------------- #
def schrodinger_np_2_level(t, state, Delta_func, Omega_func):
    Cg, Ce = state

    Delta_t = Delta_func(t)
    Omega_t = Omega_func(t)

    dCg_dt= -(1j/2) *((-Delta_t * Cg) + (Omega_t * Ce))
    dCe_dt= -(1j/2) *((Delta_t * Ce) + (Omega_t.conjugate() * Cg))

    return [dCg_dt, dCe_dt]

# ---------------------------------- #
# Parameters and initial states
# ---------------------------------- #
OmegaR0 = 5.0

Omega_func_const = lambda t: OmegaR0

initial_state = np.array([0.0, 1.0], dtype=complex)  # Initial state: Cg(0) = 0, Ce(0) = 1

t_span = (0, (6*pi/OmegaR0))
# t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# -------------------------------------------- #
# Rabi dynamics for Delta =0 and Delta = const
# -------------------------------------------- #
delta_cases = [0.0, 5.0]

fig, axes = plt.subplots(len(delta_cases), 2, figsize=(12, 9))
colors = [("navy", "darkorange"), ("purple", "pink")] 

for i, delta in enumerate(delta_cases):

    Delta_func_const = lambda t, delta=delta: delta

    # numpy ODE solution 
    sol = solve_ivp(schrodinger_np_2_level, t_span, initial_state, args=(Delta_func_const, Omega_func_const), t_eval=t_eval, rtol=1e-9, atol=1e-11)
          
    Cg_np = sol.y[0]
    Ce_np = sol.y[1]

    Pg_np = np.abs(Cg_np)**2
    Pe_np = np.abs(Ce_np)**2

    norm_np = Pg_np + Pe_np
    avg_norm_np = np.mean(norm_np)
    error_np = np.max(np.abs(norm_np - 1))

    # manual solution 
    Cg_m, Ce_m = schrodinger_manual_2_level(delta, OmegaR0, t_eval, initial_state[0], initial_state[1]) # initial_state[0] : Cg(0) = 0, initial_state[1] : Ce(0) = 1

    Pg_m = np.abs(Cg_m)**2
    Pe_m = np.abs(Ce_m)**2
    
    norm_m  = Pg_m + Pe_m
    avg_norm_m = np.mean(norm_m)
    error_m = np.max(np.abs(norm_m - 1))

    # # norm check, and err evaluation
    # print("----------------------------")
    # print(f"Delta = {delta}")
    # print("Average normalization:")
    # print("by numpy: ", np.mean(norm_np), " manually: ", np.mean(norm_m))
    # print("Maximum normalization error:")
    # print("by numpy: ", np.max(np.abs(norm_np - 1)), " manually: ", np.max(np.abs(norm_m - 1)))
    # print("----------------------------")

    # plot graphs
    Cg_color, Ce_color = colors[i]
    # numpy plot - row 0
    axes[0, i].plot(t_eval, Pg_np, label=r"$|C_g|^2$", color=Cg_color)
    axes[0, i].plot(t_eval, Pe_np, label=r"$|C_e|^2$", color=Ce_color)

    axes[0, i].set_title(rf"$\Delta={delta}$")
    axes[0, i].set_xlabel("Time")
    axes[0, i].set_ylabel("Population")
    axes[0, i].legend(loc="upper left")

    norm_text_np = (rf"$\langle P_g+P_e\rangle={avg_norm_np:.10f}$"
    "\n"
    rf"$\max|P_g+P_e-1|={error_np:.2e}$")

    axes[0, i].text(0.5, -0.30, norm_text_np, transform=axes[0, i].transAxes, ha="center", va="top", fontsize=10)
    # manual plot - row 1
    axes[1, i].plot(t_eval, Pg_m, label=r"$|C_g|^2$", color=Cg_color)
    axes[1, i].plot(t_eval, Pe_m, label=r"$|C_e|^2$", color=Ce_color)

    axes[1, i].set_title(rf"$\Delta={delta}$")
    axes[1, i].set_xlabel("Time")
    axes[1, i].set_ylabel("Population")
    axes[1, i].legend(loc="upper left")

    norm_text_m = (rf"$\langle P_g+P_e\rangle={avg_norm_m:.10f}$"
    "\n"
    rf"$\max|P_g+P_e-1|={error_m:.2e}$")
    axes[1, i].text(0.5, -0.30, norm_text_m, transform=axes[1, i].transAxes, ha="center", va="top", fontsize=10)

# Row titles
fig.text(0.5, 0.88, "solve_ivp solution",ha="center",fontsize=15)
fig.text(0.5, 0.45, "Manual solution",ha="center", fontsize=15)
# main title
fig.suptitle(rf"Rabi oscillations, $\Omega_R={OmegaR0}$"
            "\n"
            rf"C(0) = ({np.real(initial_state[0])}, {np.real(initial_state[1])})"
             ,fontsize=16,y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.95])

filename = f"Rabi_Omega_{OmegaR0}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")

plt.show()

# ----------------------------------- #
# from this part using solve_ivp only
# ----------------------------------- #
    

# ---------------------------------- #
# Scan detuning Pe Matrix
# ---------------------------------- #

delta_values = np.linspace(-5 * OmegaR0, 5 * OmegaR0, 1000)

Pe_matrix = np.zeros((len(delta_values), len(t_eval)))

for i, delta in enumerate(delta_values):

    Delta_func_const = lambda t, delta=delta: delta

    sol = solve_ivp(schrodinger_np_2_level, t_span, initial_state, args=(Delta_func_const, Omega_func_const), t_eval=t_eval, rtol=1e-9, atol=1e-11)

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
plt.title(r"$P_e$ Matrix",fontsize=15)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.figtext(0.5, 0.01, r"Some sanitiy check: in t(0): $|C_g(0)|^2 = 0, |C_e(0)|^2 = 1$ ", ha="center", fontsize=14)
plt.savefig('Pe Matrix')
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

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.text(0.5, 0.96, r"Pe x cross sections, selected $\Delta$",ha="center",fontsize=14)
plt.savefig("Pe_x_cross_sections_delta.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------- #
# y-axis cross sections
# ---------------------------------- #
selected_times = np.linspace(t_span[0], t_span[1], 5)

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

fig.suptitle(r"$P_e$ y-cross sections at selected times", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Pe_y_cross_sections_times.png", dpi=300, bbox_inches="tight")
plt.show()



# ------------------------------------- #
# 1. b. Optical Bloch Sphere  
# ------------------------------------- #

# need to solve the equation and get to each solution the Ce(t) Cg(t)
selected_deltas_bloch = [0, -5 * OmegaR0, 5 * OmegaR0 , 0.5 * OmegaR0, -0.5 * OmegaR0]

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

    Delta_func_const = lambda t, delta=delta: delta
    sol = solve_ivp(schrodinger_np_2_level, t_span, initial_state, args=(Delta_func_const, Omega_func_const), t_eval=t_eval, rtol=1e-9, atol=1e-11)

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

plt.figtext(0.5, 0.01, r"Initial states: $|C_g(0)|^2 = 0, |C_e(0)|^2 = 1$ ", ha="center", fontsize=14)
plt.savefig("Bloch_sphere_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------------------- #
# 1. c. Known solutons of two level system 
# ----------------------------------------- #


# ------------------------------------- #
# Rosen Zender
# ------------------------------------- #
Omega0 = 3.0
T = 1.0
Delta_RZ_cases = [0.0, 3.0, 9.0]

def sech(x):
    return 1 / np.cosh(x)

OmegaRZ_func = lambda t : Omega0 * sech(t / T)

# Time range
t_spanRZ = (-5*T, 5*T)
t_evalRZ = np.linspace(t_spanRZ[0], t_spanRZ[1], 3000)

# Plot Rosen Zender
fig = plt.figure(figsize=(18, 6))

for i, delta in enumerate(Delta_RZ_cases):

    # Delta(t) = Delta0 = const
    DeltaRZ_func = lambda t, delta=delta: delta

    sol = solve_ivp(schrodinger_np_2_level, t_spanRZ, initial_state, args=(DeltaRZ_func, OmegaRZ_func), t_eval=t_evalRZ, rtol=1e-9, atol=1e-11)

    Cg = sol.y[0]
    Ce = sol.y[1]

    U = (Cg.conjugate() * Ce) + (Cg * Ce.conjugate())
    V = 1j * ((Cg.conjugate() * Ce) - (Cg * Ce.conjugate()))
    W = np.abs(Ce)**2 - np.abs(Cg)**2

    # U,V should theoretically be real
    U = np.real(U)
    V = np.real(V)
    W = np.real(W)

    ax = fig.add_subplot(1, 3, i + 1, projection='3d')

    # sphere
    ax.plot_wireframe(x, y, z, alpha=0.1)

    # coordinate axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='black')
    ax.plot([0, 0], [-1, 1], [0, 0], color='black')
    ax.plot([0, 0], [0, 0], [-1, 1], color='black')

    # trajectory
    ax.plot(U, V, W, label=rf"$\Delta(t)= \Delta_0 ={delta}, \Omega(t) = \Omega_0 sech(t/T)$")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_box_aspect([1, 1, 1])

    ax.set_title(rf"$\Delta_0={delta}$", fontsize=15)
    ax.legend()

    # Custom axes
    ax.quiver(1.0-1.2, 0, 0, 1.2, 0, 0, arrow_length_ratio=0.08, color='black')
    ax.quiver(0, 1.0-1.2, 0, 0, 1.2, 0, arrow_length_ratio=0.08, color='black')
    ax.quiver(0, 0, 1.0-1.2, 0, 0, 1.2, arrow_length_ratio=0.08, color='black')

    # Labels
    ax.text(1.0 + 0.05, 0, 0, 'U', fontsize=12)
    ax.text(0, 1.0 + 0.05, 0, 'V', fontsize=12)
    ax.text(0, 0, 1.0 + 0.05, 'W', fontsize=12)

fig.suptitle(r"Rosen-Zener trajectories on the Optical Bloch Sphere"
    "\n"
    r"$\Omega(t)=\Omega_0\,\mathrm{sech}(t/T)$, "
    r"$\Delta(t)=\Delta_0$",
    fontsize=18, y=0.96)

fig.text(0.5, 0.02,
    rf"$\Omega_0={Omega0}$, $T={T:.2f}$, "
    r"Initial state: $|C_g|^2=0,\ |C_e|^2=1$",
    ha="center", fontsize=13)

plt.tight_layout(rect=[0, 0.06, 1, 0.92])

plt.savefig("Rosen Zender Solution.png", dpi=300, bbox_inches="tight")
plt.show()



# # ------------------------------------- #
# # Allen Eberly
# # ------------------------------------- #

# Omega0 = 5.0
# Delta0 = 2.5
# T = 2 * pi 
# B = 10.0

# def OmegaAE(t, T, Omega0):
#     return Omega0 * sech(t / T)

# def deltaAE(t, B, T):
#     return B * np.tanh(t / T)

# # Time range
# t_spanAE = (-5*T, 5*T)
# t_evalAE = np.linspace(-5*T, 5*T, 1000)

# def schrodingerAE(t, state, Omega0, B, T):
#     Cg, Ce = state

#     Omega_t = OmegaAE(t, T, Omega0)
#     delta_t = deltaAE(t, B, T)

#     dCg_dt= -(1j/2) *((-delta_t * Cg) + (Omega_t * Ce))
#     dCe_dt= -(1j/2) *((delta_t * Ce) + (Omega_t.conjugate() * Cg))

#     return [dCg_dt, dCe_dt]


# sol = solve_ivp(schrodingerAE, t_spanAE, initial_state, args=(Omega0, B, T), t_eval=t_evalAE, rtol=1e-9, atol=1e-11)

# Cg = sol.y[0]
# Ce = sol.y[1]

# U = (Cg.conjugate() * Ce) + (Cg * Ce.conjugate())
# V = 1j * ((Cg.conjugate() * Ce) - (Cg * Ce.conjugate()))
# W = np.abs(Ce)**2 - np.abs(Cg)**2

# # U,V should theoretically be real
# U = np.real(U)
# V = np.real(V)
# W = np.real(W)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_wireframe(x, y, z, alpha=0.1)

# # coordinate axes
# ax.plot([-1, 1], [0, 0], [0, 0], color='black')
# ax.plot([0, 0], [-1, 1], [0, 0], color='black')
# ax.plot([0, 0], [0, 0], [-1, 1], color='black')


# ax.plot(U, V, W, label=rf"$\Delta(t)= Btanh(t/T), \Omega(t) = \Omega_0sech(t/T)$")

# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])

# ax.set_box_aspect([1, 1, 1])

# ax.set_title("Trajectories on the Optical Bloch Sphere Allen Eberly Solution")
# ax.legend()

# # Custom axes
# ax.quiver(1.0-1.2, 0, 0, 1.2, 0, 0, arrow_length_ratio=0.08, color='black')
# ax.quiver(0, 1.0-1.2, 0, 0, 1.2, 0, arrow_length_ratio=0.08, color='black')
# ax.quiver(0, 0, 1.0-1.2, 0, 0, 1.2, arrow_length_ratio=0.08, color='black')

# # Labels
# ax.text(1.0 + 0.05, 0, 0, 'U', fontsize=12)
# ax.text(0, 1.0 + 0.05, 0, 'V', fontsize=12)
# ax.text(0, 0, 1.0 + 0.05, 'W', fontsize=12)

# plt.savefig("Allen Eberly Solution.png", dpi=300, bbox_inches="tight")
# plt.show()



# ------------------------------------- #
# Carroll Hioe
# ------------------------------------- #

Omega0 = 10.0
Delta_CH_cases = [0.0, 1.0, 10.0]
T = 1
B = 1

OmegaCH_func = lambda t : Omega0 * np.exp(-np.abs(t / T))

# Time range
t_spanCH = (-5*T, 5*T)
t_evalCH = np.linspace(-5*T, 5*T, 1000)

fig = plt.figure(figsize=(18,6))

for i, delta in enumerate(Delta_CH_cases):
    DeltaCH_func = lambda t, delta=delta : (delta + B * np.exp(-2 * np.abs(t / T)))

    sol = solve_ivp(schrodinger_np_2_level, t_spanCH, initial_state, args=(DeltaCH_func, OmegaCH_func), t_eval=t_evalCH, rtol=1e-9, atol=1e-11)

    Cg = sol.y[0]
    Ce = sol.y[1]

    U = (Cg.conjugate() * Ce) + (Cg * Ce.conjugate())
    V = 1j * ((Cg.conjugate() * Ce) - (Cg * Ce.conjugate()))
    W = np.abs(Ce)**2 - np.abs(Cg)**2

    # U,V should theoretically be real
    U = np.real(U)
    V = np.real(V)
    W = np.real(W)

    ax = fig.add_subplot(1, 3, i + 1, projection='3d')

    ax.plot_wireframe(x, y, z, alpha=0.1)

    # coordinate axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='black')
    ax.plot([0, 0], [-1, 1], [0, 0], color='black')
    ax.plot([0, 0], [0, 0], [-1, 1], color='black')

    ax.plot(U, V, W, label=(rf"$\Delta_0={delta}$"))

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_box_aspect([1, 1, 1])

    ax.legend(loc="upper left")

    # Custom axes
    ax.quiver(1.0-1.2, 0, 0, 1.2, 0, 0, arrow_length_ratio=0.08, color='black')
    ax.quiver(0, 1.0-1.2, 0, 0, 1.2, 0, arrow_length_ratio=0.08, color='black')
    ax.quiver(0, 0, 1.0-1.2, 0, 0, 1.2, arrow_length_ratio=0.08, color='black')

    # Labels
    ax.text(1.0 + 0.05, 0, 0, 'U', fontsize=12)
    ax.text(0, 1.0 + 0.05, 0, 'V', fontsize=12)
    ax.text(0, 0, 1.0 + 0.05, 'W', fontsize=12)

fig.suptitle(
    r"Carroll-Hioe trajectories on the Optical Bloch Sphere"
    "\n"
    r"$\Omega(t)=\Omega_0 e^{-|t/T|}$, "
    r"$\Delta(t)=\Delta_0 + B e^{-2|t/T|}$", fontsize=18, y=0.96)

fig.text(0.5, 0.02,
    rf"$\Omega_0={Omega0}$, $T={T:.2f}$, $B={B}$, "
    r"Initial state: $|C_g|^2=0,\ |C_e|^2=1$", ha="center", fontsize=13)

plt.tight_layout(rect=[0, 0.06, 1, 0.92])
plt.savefig("Carroll Hioe Solution.png", dpi=300, bbox_inches="tight")
plt.show()



# ------------------------------------- #
# 1.d. Bonus find the requirements for full population transfer
# if i have time, for RZ there's analytic solution, but for CH numeric calculation needed
# ------------------------------------- #



# ---------------------------------- #
# Part 2 - Three level system  
# ---------------------------------- #

# ---------------------------------- #
# 2.a. Define the ODEs using numpy
# ---------------------------------- #

# accourding to the hemiltonian 2.a.
def schrodinger_3_level(t, state, Delta_func, delta_func, Omega12_func, Omega23_func):
    C1, C2, C3 = state

    Delta_t = Delta_func(t)
    delta_t = delta_func(t)
    Omega12_t = Omega12_func(t)
    Omega23_t = Omega23_func(t)

    dC1_dt= -(1j/2) *(Omega12_t * C2)
    dC2_dt= -(1j/2) *((Omega12_t.conjugate() * C1) + (Delta_t * C2)+(Omega23_t * C3))
    dC3_dt= -(1j/2) *((Omega23_t.conjugate()* C2) + (delta_t * C3))

    return [dC1_dt, dC2_dt, dC3_dt]

# ---------------------------------- #
# 2.b.- 2.c. Rabi-like solution
# ---------------------------------- #
Omega12_cases = Omega23_cases = [5.0, 5.0, 5.0, 5.0]
Delta0_cases = [0.0, 0.0, 2 * Omega12_cases[1], 2 * Omega12_cases[1]]
delta0_cases = [0.0, 0.0, 4 * Omega12_cases[1], 4 * Omega12_cases[1]]

# define initaial state cases:
initial_state_cases = np.array([(1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0)], dtype=complex)

# Time 
t_span = (0, (6*pi/Omega12_cases[0]))
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# Plot def
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

colors = ["navy", "darkorange", "purple"]
# colors = [("navy", "darkorange"), ("purple", "pink")] 

for i in range(len(Omega12_cases)):
    Omega12_0 = Omega12_cases[i]
    Omega23_0 = Omega23_cases[i]
    Delta0 = Delta0_cases[i]
    delta0 = delta0_cases[i]

     # Delta(t), delta(t), Omega12(t), Omega23(t) are all const
    Omega12 = lambda t, Omega12_0=Omega12_0: Omega12_0
    Omega23 = lambda t, Omega23_0=Omega23_0: Omega23_0
    Delta = lambda t, Delta0=Delta0: Delta0
    delta = lambda t, delta0=delta0: delta0

    sol = solve_ivp(schrodinger_3_level, t_span, initial_state_cases[i], args=(Delta, delta, Omega12, Omega23), t_eval=t_eval, rtol=1e-9, atol=1e-11)

    C1, C2, C3 = sol.y[0], sol.y[1], sol.y[2]
    P1, P2, P3 = np.abs(C1)**2, np.abs(C2)**2, np.abs(C3)**2
    norm = P1 + P2 + P3
    avg_norm = np.mean(norm)
    error = np.max(np.abs(norm - 1))

   # ---------------------------------- #
    # Plot
    # ---------------------------------- #
    ax = axes[i]

    ax.plot(t_eval, P1, label=r"$|C_1|^2$", color=colors[0])
    ax.plot(t_eval, P2, label=r"$|C_2|^2$", color=colors[1])
    ax.plot(t_eval, P3, label=r"$|C_3|^2$", color=colors[2])

    ax.set_xlabel("Time")
    ax.set_ylabel("Population")

    ax.set_ylim(-0.05, 1.05)

    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # initial state for title
    c10, c20, c30 = np.real(initial_state_cases[i])

    ax.set_title(rf"$\Omega_{{12}}=\Omega_{{23}}={Omega12_0}$"
        "\n"
        rf"$\Delta={Delta0},\ \delta={delta0}$"
        "\n"
        rf"$C(0)=({c10},{c20},{c30})$")

    norm_text = (rf"$\langle P_1+P_2+P_3\rangle={avg_norm:.10f}$"
        "\n"
        rf"$\max|P_1+P_2+P_3-1|={error:.2e}$")

    ax.text(0.5, -0.30, norm_text, transform=ax.transAxes, ha="center", va="top", fontsize=10)

fig.suptitle(r"Three-level system dynamics"
             "\n"
             "Rabi like solution", fontsize=18, y=0.99)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("Three_level_Rabi_like.png", dpi=300, bbox_inches="tight")
plt.show()



# ----------------------------------- #
# 2.d. Adiabatic Elimination solution
# ----------------------------------- #

Omega12_cases = [5.0, 5.0, 5.0, 5.0]
Omega23_cases = [5.0, 5.0, 5.0, 5.0]
Delta0_cases = [10 * Omega12_cases[1], 10 * Omega12_cases[1], 20 * Omega12_cases[1], 20 * Omega12_cases[1]]
delta0_cases = [0.0, 0.0, 0.0, 0.0]


# Plot def
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for i in range(len(Omega12_cases)):
    Omega12_0 = Omega12_cases[i]
    Omega23_0 = Omega23_cases[i]
    Delta0 = Delta0_cases[i]
    delta0 = delta0_cases[i]

    Omega_eff = Omega12_0 * Omega23_0 / Delta0
    # Time 
    t_span = (0, (3*pi/Omega_eff))
    t_eval = np.linspace(t_span[0], t_span[1], 3000)    

     # Delta(t), delta(t), Omega12(t), Omega23(t) are all const
    Omega12 = lambda t, Omega12_0=Omega12_0: Omega12_0
    Omega23 = lambda t, Omega23_0=Omega23_0: Omega23_0
    Delta = lambda t, Delta0=Delta0: Delta0
    delta = lambda t, delta0=delta0: delta0

    sol = solve_ivp(schrodinger_3_level, t_span, initial_state_cases[i], args=(Delta, delta, Omega12, Omega23), t_eval=t_eval, rtol=1e-9, atol=1e-11)

    C1, C2, C3 = sol.y[0], sol.y[1], sol.y[2]
    P1, P2, P3 = np.abs(C1)**2, np.abs(C2)**2, np.abs(C3)**2
    norm = P1 + P2 + P3
    avg_norm = np.mean(norm)
    error = np.max(np.abs(norm - 1))

    # ---------------------------------- #
    # Plot
    # ---------------------------------- #
    ax = axes[i]

    ax.plot(t_eval, P1, label=r"$|C_1|^2$", color=colors[0])
    ax.plot(t_eval, P2, label=r"$|C_2|^2$", color=colors[1])
    ax.plot(t_eval, P3, label=r"$|C_3|^2$", color=colors[2])

    ax.set_xlabel("Time")
    ax.set_ylabel("Population")

    ax.set_ylim(-0.05, 1.05)

    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # initial state for title
    c10, c20, c30 = np.real(initial_state_cases[i])

    ax.set_title(rf"$\Omega_{{12}}=\Omega_{{23}}={Omega12_0}$"
        "\n"
        rf"$\Delta={Delta0},\ \delta={delta0}$"
        "\n"
        rf"$C(0)=({c10},{c20},{c30})$")

    norm_text = (rf"$\langle P_1+P_2+P_3\rangle={avg_norm:.10f}$"
        "\n"
        rf"$\max|P_1+P_2+P_3-1|={error:.2e}$")

    ax.text(0.5, -0.30, norm_text, transform=ax.transAxes, ha="center", va="top", fontsize=10)

fig.suptitle(r"Three-level system dynamics"
             "\n"
             "Adiabatic Elimination solution full H", fontsize=18, y=0.99)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("Three_level_AE.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------- #
# 2.e. solving via Heff
# ---------------------------------- #

Omega_eff = Omega12_0 * Omega23_0 / Delta0