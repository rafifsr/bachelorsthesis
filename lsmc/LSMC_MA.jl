import Pkg; Pkg.activate(@__DIR__)
using DifferentialEquations, LinearAlgebra, Statistics, Random
using StatsPlots, Plots, Distributions, KernelDensity, StatsBase, CSV, DataFrames
using LaTeXStrings, Measures

# === Load Experimental Data ===
df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
dfs = hcat(df.Xa, df.Xi, df.N, df.S, df.FG, df.MA)
u0 = dfs[1, :]
t_exp = df.time
T = maximum(t_exp)
tspan = (0.0, T)
dt = 0.01
tsteps = collect(0.0:dt:T)
Nt = length(tsteps)

# === Define ODE Model ===
function f!(du, u, p, t)
    # Unpack state and parameters
    μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2, σxa, σxi, σs, σfg, σp = p
    Xact, Xinact, N, Suc, FruGlu, P = u

    # Ensure non-negative values
    ϵ = 1e-8  # Small positive value to avoid division by zero
    Xact_safe = max(Xact, ϵ)
    Xtot_safe = max(Xact + Xinact, ϵ)
    FruGlu_safe = max(FruGlu, ϵ)
    Suc_safe = max(Suc, ϵ)

    # Algebraic equations
    Xtot = Xact + Xinact
    N_int = 0.08 * N
    ratio = Xinact / Xact_safe
    expo_term = (ratio - ϕ) / χacc

    μ = μmax * FruGlu_safe / (FruGlu_safe + KFG + ϵ) * (N / (N+ KN + ϵ))
    μ2 = μ2max * FruGlu_safe / (FruGlu_safe + KFG2 + ϵ) * (1 - exp(expo_term)) * KIN / (KIN + N + ϵ)
    qsplit = qsplit_max * Suc_safe / (Suc_safe + Ksuc + ϵ)
    qp = qpmax * FruGlu_safe / (FruGlu_safe + KPFG + ϵ) *
         (KIP / (KIP + N_int / Xtot_safe + ϵ)) * KIN / (KIN + N + ϵ)

    du[1] = μ * Xact
    du[2] = μ2 * Xact
    du[3] = - (μ / YXa_N) * Xact
    du[4] = - qsplit * Xact
    du[5] = (qsplit - μ / YXa_S - μ2 / YXi_S - qp / YP_S) * Xact
    du[6] = qp * Xact
end

# === Define Noise Function ===
function noise!(du, u, p, t)
    Xact, Xinact, N, Suc, FruGlu, P = u
    μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2, σxa, σxi, σs, σfg, σp = p
    du[1] = σxa * FruGlu / (FruGlu + (KFG)) * Xact
    du[2] = σxi * FruGlu / (FruGlu + (KFG2)) * Xinact
    du[3] = 0.0
    du[4] = 0.0
    # du[3] = (σxa / YXa_N) * FruGlu / (FruGlu + (KFG)) * Xact
    # du[4] = σs * Suc / (Suc + Ksuc) * Xact
    du[5] = σfg * FruGlu / (FruGlu + Ksuc) * (Xact + Xinact)
    du[6] = σp * FruGlu / (FruGlu + (KPFG)) * Xact
end

# === Parameters ===
params = [
    0.125,  # 1. μmax
    0.147,  # 2. KFG
    3.8e-5,  # 3. KN
    0.531,  # 4. YXa_S
    0.799,  # 5. YXi_S
    9.428,  # 6. YXa_N
    0.508,  # 7. YP_S
    1.56,  # 7. ϕ
    0.3,  # 8. χacc
    0.125,  # 9. μ2max
    1.985,  # 10. qsplit_max
    0.00321,  # 11. Ksuc
    0.095,  # 12. qpmax
    1.5,  # 13. KIP
    1.5e-3,  # 14. KIN
    0.0175,  # 15. KPFG
    3.277,  # 16. KFG2
    0.05,   # 17. σxa
    0.05,   # 18. σxi
    0.05,   # 19. σn
    0.05,   # 20. σs
    0.05,   # 21. σfg
    0.05    # 22. σp
]

# === Solve the ODE ===
odeprob = ODEProblem(f!, u0, tspan, params)
odesol = solve(odeprob, Rosenbrock23(), saveat=tsteps, abstol=1e-8, reltol=1e-6)

# === Extract ODE solution ===
Xa_ode = odesol[1, :]
Xi_ode = odesol[2, :]
N_ode  = odesol[3, :]
Suc_ode = odesol[4, :]
FG_ode = odesol[5, :]
P_ode = odesol[6, :]

# === Define the non-negative callback and the SDE Problem ===
function project!(integrator)
    integrator.u .= max.(integrator.u, 0.0)
end
proj_cb = DiscreteCallback((u,t,integrator) -> true, project!; save_positions = (false, false))

sdeprob = SDEProblem(f!, noise!, u0, tspan, params)

# === Simulate the scenarios ===
M = 1000  # Number of simulations

ensemble_prob = EnsembleProblem(sdeprob, prob_func = (prob, i, repeat) -> remake(prob, u0 = u0))
ensemble_sol = solve(ensemble_prob, EM(), EnsembleThreads(), trajectories=M, dt=dt,
                     saveat=tsteps, callback=proj_cb)

# Extract results
Xact_mat = hcat([sol[1, :] for sol in ensemble_sol]...)
Xinact_mat = hcat([sol[2, :] for sol in ensemble_sol]...)
N_mat = hcat([sol[3, :] for sol in ensemble_sol]...)
Suc_mat = hcat([sol[4, :] for sol in ensemble_sol]...)
FruGlu_mat = hcat([sol[5, :] for sol in ensemble_sol]...)
P_mat = hcat([sol[6, :] for sol in ensemble_sol]...)

# === Laguerre basis functions up to degree 3 ===
d = 3  # Degree of the Laguerre polynomial

function laguerre_design_matrix(y::Vector{Float64}, d::Int)
    Φ = zeros(length(y), d + 1)
    for i in 1:length(y)
        Φ[i,1] = 1.0
        if d >= 1
            Φ[i,2] = 1 - y[i]
        end
        if d >= 2
            Φ[i,3] = 1 - 2*y[i] + 0.5*y[i]^2
        end
        if d >= 3
            Φ[i,4] = 1 - 3*y[i] + 1.5*y[i]^2 - (1/6)*y[i]^3
        end
    end
    return Φ
end

# === Reward function ===
reward(s) = -(s-thresh)^2

# === Initialize vectors and matrices ===
β_matrix = zeros(Nt, d + 1) # Matrix to store the regression coefficients on every time step
thresh = 0.05 *  (u0[4] + u0[5]) # Threshold for stopping condition: 5% of initial substrates (Suc + FruGlu)

# === Calculate the reward at all time steps ===
rewards = reward.(FruGlu_mat)

# === Filter the paths that have not reached the threshold at the last time step (in-the-money) ===
valid_paths = findall(i -> FruGlu_mat[Nt, i] ≤ thresh, 1:M)
if isempty(valid_paths)
    error("No valid paths found at the last time step.")
end

τ = fill(Nt, length(valid_paths))  # Stopping times for valid paths

# === Prepare the matrices for backward induction ===
# Only keep the valid paths for the backward induction
s_valid = FruGlu_mat[:, valid_paths] 
rewards_valid = rewards[:, valid_paths]

# === The backward induction ===
for n in (Nt-1):-1:2
    s_now = s_valid[n, :]  # Current state for valid paths
    reward_now = rewards_valid[n, :]  # Rewards at the current time step
    reward_future = rewards_valid[n+1, :]  # Rewards at the next time step
    
    # Regression and saving the coefficients
    ϕ = laguerre_design_matrix(s_now, d)
    β = ϕ \ reward_future  # Solve the linear system to find coefficients
    # β_matrix[n, :] = β  # Store the coefficients for this time step
    E = ϕ * β  # Calculate the expected reward for the valid paths

    for (k,i) in enumerate(valid_paths)
        if E[k] ≤ reward_now[k]  # If the expected reward is less than or equal to the current reward
            τ[k] = n  # Update the stopping time for this path
            rewards_valid[n, k] = reward_now[k]  # Keep the current reward
        else
            rewards_valid[n, k] = E[k]  # Otherwise, use the expected reward
        end
    end
end

# Store the raw index at which each path exercised
stop_idx   = copy(τ)          # length = M, integer ∈ 1:Nt

# Convert the index to an actual time
stop_time  = tsteps[stop_idx]         # same length as stop_idx, Float64

# Plot the histogram
histogram(stop_time;
          bins  = :auto,              # let StatsPlots pick a sensible bin count
          xlabel = "Stopping time t",
          ylabel = "Number of paths",
          title  = "Distribution of optimal stopping times")