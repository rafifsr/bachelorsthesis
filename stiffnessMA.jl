using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, DataFrames, CSV, Plots, Measures, LaTeXStrings, ForwardDiff, LinearAlgebra

# Load Experimental Data
df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
dfs = hcat(df.Xa, df.Xi, df.N, df.S, df.FG, df.MA)

# Identify parameters and initial conditions
u0 = dfs[1, :]
t = df.time
tspan = (t[1], t[end])
dt = 0.01

# Define ODE Model
function f!(du, u, p, t)
    # Unpack state and parameters
    μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2 = p
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

# === Parameters Maschmeier ===
params_MA = [
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
    28.188,  # 12. qpmax
    1.47e-4,  # 13. KIP
    1.47e-4,  # 14. KIN
    0.0175,  # 15. KPFG
    3.277  # 16. KFG2
]

# === Correct Parameters ===
params_new = [
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
    1.5e-1,  # 13. KIP
    1.5e-2,  # 14. KIN
    0.0175,  # 15. KPFG
    3.277  # 16. KFG2
]

# Compute stiffness over time
# This function computes the stiffness ratio over time for the given ODE problem.
function stiffness_over_time(u0, p)
    prob = ODEProblem(f!, u0, (t[1], t[end]), p)
    sol = solve(prob, Rosenbrock23(), saveat=0.2)
    ratios = Float64[]

    for u in sol.u
        J = ForwardDiff.jacobian(u -> begin
            du = similar(u)
            f!(du, u, p, 0.0)
            return du
        end, u)

        λ = eigvals(J)
        λ_abs = abs.(λ[abs.(λ) .> 1e-10])
        ratio = maximum(λ_abs) / minimum(λ_abs)
        push!(ratios, ratio)
    end
    return sol.t, ratios
end

t1, ratios1 = stiffness_over_time(u0, params_MA)
t2, ratios2 = stiffness_over_time(u0, params_new)

p = plot(yscale=:log10, xlabel="Time / h", ylabel="Stiffness Ratio", xlims=(t[1], t[end]), fontfamily="Computer Modern",
        legend = true, legendfont = 13, tickfont = 11, guidefont = 16, margins = 5mm, size = (900,600))
plot!(t1, ratios1, label = "Maschmeier (2024)", lw = 3)
plot!(t2, ratios2, label = "New Parameters", lw = 3, linestyle = :dash)
savefig(p, "Figures/stiffness_ratios.pdf")
display(p)