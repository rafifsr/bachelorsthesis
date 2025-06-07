using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO

function odes!(du, u, p, t)
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

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5]
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5())

# Generate synthetic data
t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

# Define the loss function
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t, data),
                                     Optimization.AutoForwardDiff(),
                                     maxiters = 10000, verbose = false)

# Define the optimization problem
optprob = Optimization.OptimizationProblem(cost_function, [1.42])
optsol = solve(optprob, BFGS())
newprob = remake(prob, p = optsol.u)
newsol = solve(newprob, Tsit5())
plot(sol)
plot!(newsol, linestyle = :dash, lw=2)