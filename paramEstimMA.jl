using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO, OptimizationOptimJL, SciMLSensitivity
using CSV, DataFrames

# ------------------------------
# 1. Load Experimental Data
# ------------------------------
df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
df_Xa = df[!, [:time, :Xa]]
df_Xi = df[!, [:time, :Xi]]
df_N = df[!, [:time, :N]]
df_Suc = df[!, [:time, :S]]
df_FruGlu = df[!, [:time, :FG]]
df_MA = df[!, [:time, :MA]]
dfs = [df_Xa, df_Xi, df_N, df_Suc, df_FruGlu, df_MA]

# ------------------------------
# 2. Define ODE Model
# ------------------------------
function f!(du, u, p, t)
    (μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc,
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

# ------------------------------
# 3. Prepare Time Grid, Parameters, and Initial Condition
# ------------------------------
times = df.time
t = collect(times)

# Assume initial condition from first available values
u0 = [df_Xa.Xa[1], df_Xi.Xi[1], df_N.N[1], df_Suc.S[1], df_FruGlu.FG[1], df_MA.MA[1]]

# Guessed parameters
p = [
    0.125,  # μmax
    0.147,  # KFG
    0.000038,  # KN
    0.531,  # YXa_S
    0.799,  # YXi_S
    9.428,  # YXa_N
    0.508,  # YP_S
    1.56,  # ϕ
    0.3,  # χacc
    0.125,  # μ2max
    1.985,  # qsplit_max
    0.00321,  # Ksuc
    0.28188,  # qpmax
    0.000147,  # KIP
    0.000147,  # KIN
    0.0175,  # KPFG
    3.277  # KFG2
]
# ------------------------------
# 4. Define Loss Function and Optimization Problem
# ------------------------------
prob = ODEProblem(f!, u0, (0.0, maximum(t)), p)

cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t, df),
                                     Optimization.AutoForwardDiff(),
                                     maxiters = 10000, verbose = false)

optprob = Optimization.OptimizationProblem(cost_function, p)
optsol = solve(optprob, BFGS())
newprob = remake(prob, p = optsol.u)
newsol = solve(newprob, Tsit5())
println("Optimized parameters: ", optsol.u)

# ------------------------------
# 5. Plot FruGlu Results (experimental data vs guessed parameters vs optimized)
# ------------------------------
plot(df.time, df.FG, label = "Experimental FruGlu", xlabel = "Time", ylabel = "FruGlu", title = "FruGlu Concentration Over Time")
plot!(newsol.t, newsol[5, :], label = "Optimized FruGlu", linestyle = :dash)