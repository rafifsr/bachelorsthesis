using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO, OptimizationOptimJL, SciMLSensitivity
using CSV, DataFrames
using DiffEqBayes, OrdinaryDiffEq, Distributions, StatsPlots, BenchmarkTools, TransformVariables, DynamicHMC

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
    μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2 = p

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
params = [
    0.125,  # μmax
    0.147,  # KFG
    3.8e-5,  # KN
    0.531,  # YXa_S
    0.799,  # YXi_S
    9.428,  # YXa_N
    0.508,  # YP_S
    1.56,  # ϕ
    0.3,  # χacc
    0.125,  # μ2max
    1.985,  # qsplit_max
    0.00321,  # Ksuc
    2.8188,  # qpmax
    1.5e-4,  # KIP
    1.5e-4,  # KIN
    0.0175,  # KPFG
    3.277  # KFG2
]

# ------------------------------
# 4. Define Loss Function and Optimization Problem
# ------------------------------
prob = ODEProblem(f!, u0, (0.0, maximum(t)), params)
sol = solve(prob, Tsit5())

cost_function = build_loss_objective(prob, Tsit5(), 
                                     L2Loss(t, df, differ_weight = 0.3, data_weight = 0.7),
                                     Optimization.AutoForwardDiff(),
                                     maxiters = 10000, verbose = false)

optprob = Optimization.OptimizationProblem(cost_function, params)
optsol = solve(optprob, BFGS())
newprob = remake(prob, p = optsol.u)
newsol = solve(newprob, Tsit5())
println("Optimized parameters: ", optsol.u)

# ------------------------------
# 5. Plot Results (experimental data vs guessed parameters vs optimized)
# ------------------------------
# Create subplots for each variable
plot_layout = @layout [a b; c d; e f]
p = plot(layout = plot_layout, size = (1200, 800), fontfamily = "Computer Modern")

# Plot each variable
scatter!(p[1], df.time, df.Xa, label = "Experimental Xa", xlabel = "Time", ylabel = "Xa", legend = :bottomright)
plot!(p[1], newsol.t, newsol[1, :], label = "Optimized Xa", linestyle = :dash)
plot!(p[1], sol.t, sol[1, :], label = "Guessed Xa", linestyle = :dot, xlims = (0, 40), ylims = (-0.1, 10))

scatter!(p[2], df.time, df.Xi, label = "Experimental Xi", xlabel = "Time", ylabel = "Xi")
plot!(p[2], newsol.t, newsol[2, :], label = "Optimized Xi", linestyle = :dash)
plot!(p[2], sol.t, sol[2, :], label = "Guessed Xi", linestyle = :dot, xlims = (0, 40), ylims = (-0.1, 14))

scatter!(p[3], df.time, df.N, label = "Experimental N", xlabel = "Time", ylabel = "N")
plot!(p[3], newsol.t, newsol[3, :], label = "Optimized N", linestyle = :dash)
plot!(p[3], sol.t, sol[3, :], label = "Guessed N", linestyle = :dot, xlims = (0, 40), ylims = (-0.01, 0.8))

scatter!(p[4], df.time, df.S, label = "Experimental Suc", xlabel = "Time", ylabel = "Suc")
plot!(p[4], newsol.t, newsol[4, :], label = "Optimized Suc", linestyle = :dash)
plot!(p[4], sol.t, sol[4, :], label = "Guessed Suc", linestyle = :dot, xlims = (0, 40), ylims = (-100, 65))

scatter!(p[5], df.time, df.FG, label = "Experimental FruGlu", xlabel = "Time", ylabel = "FruGlu")
plot!(p[5], newsol.t, newsol[5, :], label = "Optimized FruGlu", linestyle = :dash)
plot!(p[5], sol.t, sol[5, :], label = "Guessed FruGlu", linestyle = :dot, xlims = (0, 40), ylims = (-1, 100))

scatter!(p[6], df.time, df.MA, label = "Experimental MA", xlabel = "Time", ylabel = "MA")
plot!(p[6], newsol.t, newsol[6, :], label = "Optimized MA", linestyle = :dash)
plot!(p[6], sol.t, sol[6, :], label = "Guessed MA", linestyle = :dot, xlims = (0, 40), ylims = (-0.1, 25))

# Display the plot
display(p)

# Save the plot to a file
savefig(p, "Figures/param_estim_ma.pdf")