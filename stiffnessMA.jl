using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, DataFrames, CSV, Plots, Measures, LaTeXStrings, ForwardDiff, LinearAlgebra

# Load Experimental Data
df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
dfs = hcat(df.Xa, df.Xi, df.N, df.S, df.FG, df.MA)
df_n = CSV.read("datasetsMA/n.csv", DataFrame)
df_fruglu = CSV.read("datasetsMA/fruglu.csv", DataFrame)
df_ma = CSV.read("datasetsMA/malicacid.csv", DataFrame)
df_suc = CSV.read("datasetsMA/suc.csv", DataFrame)
df_xa = CSV.read("datasetsMA/xa.csv", DataFrame)
df_xi = CSV.read("datasetsMA/xi.csv", DataFrame)

# Identify parameters and initial conditions
u0 = dfs[1, :]
t = df.time
tspan = (t[1], t[end])
dt = 0.1

# Define ODE Model
function f!(du, u, p, t)
    # Unpack state and parameters
    μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2, σxa, σxi, σn, σs, σfg, σp = p
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

function noise!(du, u, p, t)
    Xact, Xinact, N, Suc, FruGlu, P = u
    μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2, σxa, σxi, σn, σs, σfg, σp = p
    du[1] = σxa * FruGlu / (FruGlu + (KFG)) * Xact
    du[2] = σxi * FruGlu / (FruGlu + (KFG2)) * Xact
    du[3] = σn * N / (N + (KN)) * Xact
    du[4] = σs * Suc / (Suc + (Ksuc)) * Xact
    du[5] = σfg * FruGlu / (FruGlu + (Ksuc + KFG + KFG2 + KPFG)) * Xact
    du[6] = σp * FruGlu / (FruGlu + (KPFG)) * Xact
end

# === Parameters Maschmeier ===
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
    28.188,  # 12. qpmax
    1.47e-4,  # 13. KIP
    1.47e-4,  # 14. KIN
    0.0175,  # 15. KPFG
    3.277,  # 16. KFG2
    5e-2,   # 17. σxa
    5e-2,   # 18. σxi
    1e-2,   # 19. σn
    5e-2,   # 20. σs
    5e-2,   # 21. σfg
    5e-2    # 22. σp
]

# === Correct Parameters ===
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
    1.5e-1,  # 13. KIP
    1.5e-2,  # 14. KIN
    0.0175,  # 15. KPFG
    3.277,  # 16. KFG2
    5e-2,   # 17. σxa
    5e-2,   # 18. σxi
    1e-2,   # 19. σn
    5e-2,   # 20. σs
    5e-2,   # 21. σfg
    5e-2    # 22. σp
]

# === Solve the ODE ===
prob = ODEProblem(f!, u0, tspan, params)
sol = solve(prob, Rosenbrock23(), saveat=dt, abstol=1e-8, reltol=1e-6)

# Compute Jacobian and stiffness ratio at each saved step
times = sol.t
ratios = Float64[]

for u in sol.u
    J = ForwardDiff.jacobian(u -> begin
        du = similar(u)
        f!(du, u, params, 0.0)
        return du
    end, u)

    λ = eigvals(J)
    λ_abs = abs.(λ[abs.(λ) .> 1e-10])  # avoid very small values
    ratio = maximum(λ_abs) / minimum(λ_abs)
    push!(ratios, ratio)
end

# Plot
plot(times, ratios, yscale=:log10, xlabel="Time / h", ylabel="Stiffness Ratio",
     lw = 2, xlims=(0, maximum(times)), fontfamily="Computer Modern")