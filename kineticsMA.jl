using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, Random, Distributions, Statistics

# === Parameters ===
params = (
    μmax = 0.125, 
    KFG = 0.147, 
    KN = 0.000038, 
    YXa_S = 0.531, 
    YXi_S = 0.799, 
    YXa_N = 9.428, 
    YP_S = 0.508, 
    ϕ = 1.56, 
    χacc = 0.3,
    μ2max = 0.125, 
    qsplit_max = 1.985, 
    Ksuc = 0.00321, 
    qpmax = 0.28188, 
    KIP = 0.000147, # check again
    KIN = 0.000147, # check again  
    KPFG = 0.0175, 
    KFG2 = 3.277
)

# === ODE RHS ===
function odes!(du, u, p, t)
    (; μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc,
       μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2) = p

    Xact, Xinact, N, Suc, FruGlu, P = u

    # Algebraic equations
    N_int = 0.08 * N
    Xtot = Xact + Xinact
    ratio = Xinact / Xact
    expo_term = (ratio - ϕ) / χacc

    μ = μmax * FruGlu / (FruGlu + KFG) * (N / (N + KN))
    μ2 = μ2max * FruGlu / (FruGlu + KFG2) * (1 - exp(expo_term)) * KIN / (KIN + N)
    qsplit = qsplit_max * (Suc / (Suc + Ksuc))
    qp = qpmax * FruGlu / (FruGlu + KPFG) * (KIP / (KIP + N_int/Xtot)) * KIN / (KIN + N)

    du[1] = μ * Xact
    du[2] = μ2 * Xact
    du[3] = - (μ / YXa_N) * Xact
    du[4] = - qsplit * Xact
    du[5] = (qsplit - μ / YXa_S - μ2 / YXi_S - qp / YP_S) * Xact
    du[6] = qp * Xact
end

# === Initial Conditions ===
u0 = [2.0, 0.0, 0.1, 65, 10.0, 0.0]  # [Xact, Xinact, N, Suc, FruGlu, P]
T = 40.0  # Total time for the simulation
tspan = (0.0, T)
dt = 0.01
tsteps = 0.0:dt:T  # Time steps for the solution

# === Termination Callback ===
function stop_condition(u, t, integrator)
    Suc = u[4]
    FruGlu = u[5]
    return Suc + FruGlu
end

terminate_cb = ContinuousCallback(stop_condition, terminate!, rootfind=true)

# === Solve the ODE ===
prob = ODEProblem(odes!, u0, tspan, params)
sol = solve(prob, Tsit5())#, callback=terminate_cb)

# Save the solutions for each state variable
Xa_sol = sol[1, :]
Xin_sol = sol[2, :]
N_sol = sol[3, :]
Suc_sol = sol[4, :]
FruGlu_sol = sol[5, :]
P_sol = sol[6, :]

# === Plotting ===
using Plots
plot(sol.t, FruGlu_sol, 
     xlabel = "Time (h)", 
     ylabel = "FruGlu Concentration (g/L)",
     legend = false,
     ylims = (0, 60),
     xlims = (0, T),
     lw = 2)

# plot(sol, xlabel="Time (h)", ylabel="Concentration (g/L)", lw=2,
#      label=["Xa" "Xin" "N" "Suc" "FruGlu" "MA"], legend=:right, fontfamily="Computer Modern",
#      xlims=(0, T) #, ylims=(0, u0[4])
#      )
