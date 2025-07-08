using Pkg
Pkg.activate(@__DIR__)
using DataFrames, CSV, DifferentialEquations, Plots
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO, RecursiveArrayTools, DiffEqParamEstim

# Load Experimental Data
df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
dfs = hcat(df.Xa, df.Xi, df.N, df.S, df.FG, df.MA)
u0 = dfs[1, :]
t = df.time
tspan = (t[1], t[end])
println("Data loaded successfully. Number of rows: ", nrow(df))

# Define ODE Model
function f!(du, u, p, t)

    # Unpack state and parameters
    qpmax, KIP, KIN = p
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
    expo_term = (ratio - 1.56) / 0.3

    μ = 0.125 * FruGlu_safe / (FruGlu_safe + 0.147 + ϵ) * (N / (N+ 3.8e-5 + ϵ))
    μ2 = 0.125 * FruGlu_safe / (FruGlu_safe + 3.277 + ϵ) * (1 - exp(expo_term)) * KIN / (KIN + N + ϵ)
    qsplit = 1.985 * Suc_safe / (Suc_safe + 0.00321 + ϵ)
    qp = qpmax * FruGlu_safe / (FruGlu_safe + 0.0175 + ϵ) * (KIP / (KIP + N_int / Xtot_safe + ϵ)) * KIN / (KIN + N + ϵ)

    du[1] = μ * Xact
    du[2] = μ2 * Xact
    du[3] = - (μ / 9.428) * Xact
    du[4] = - qsplit * Xact
    du[5] = (qsplit - μ / 0.531 - μ2 / 0.799 - qp / 0.508) * Xact
    du[6] = qp * Xact
end

# 3. Guessed Parameters 
params = [
    0.095,  # qpmax
    1.47e-1,  # KIP
    1.47e-2,  # KIN
]

# Solve the ODE
prob = ODEProblem(f!, u0, tspan, params)
sol = solve(prob, Rodas5P(), saveat=0.1, abstol=1e-8, reltol=1e-6)

# Optimization Problem
cost_function = build_loss_objective(prob, Rodas5P(), L2Loss(t, dfs), saveat=0.1, Optimization.AutoForwardDiff(),
                                     maxiters = 10_000, abstol=1e-8, reltol=1e-6, verbose=true)
optprob = Optimization.OptimizationProblem(cost_function, params)
optsol = solve(optprob, BFGS())
println("Optimization completed. Optimal parameters: ", optsol.u)

# Solve ODE with optimized parameters
newprob = ODEProblem(f!, u0, tspan, optsol.u)
newsol = solve(newprob, Rodas5P(), saveat=0.1, abstol=1e-8, reltol=1e-6)

# Visualize the results
plot_layout = @layout [a b; c d; e f]
p = plot(layout = plot_layout, size = (1200, 800), fontfamily = "Computer Modern")

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
plot!(p[4], sol.t, sol[4, :], label = "Guessed Suc", linestyle = :dot, xlims = (0, 40), ylims = (-1, 65))

scatter!(p[5], df.time, df.FG, label = "Experimental FruGlu", xlabel = "Time", ylabel = "FruGlu")
plot!(p[5], newsol.t, newsol[5, :], label = "Optimized FruGlu", linestyle = :dash)
plot!(p[5], sol.t, sol[5, :], label = "Guessed FruGlu", linestyle = :dot, xlims = (0, 40), ylims = (-1, 100))

scatter!(p[6], df.time, df.MA, label = "Experimental MA", xlabel = "Time", ylabel = "MA")
plot!(p[6], newsol.t, newsol[6, :], label = "Optimized MA", linestyle = :dash)
plot!(p[6], sol.t, sol[6, :], label = "Guessed MA", linestyle = :dot, xlims = (0, 40), ylims = (-0.1, 25))

# Display the plot
display(p)