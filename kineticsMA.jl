using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, Random, Distributions, Statistics, DataFrames, CSV

# Load Experimental Data
df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
dfs = hcat(df.Xa, df.Xi, df.N, df.S, df.FG, df.MA)
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

# === Extend to SDE ===
function project!(integrator)
    integrator.u .= max.(integrator.u, 0.0)
end
proj_cb = DiscreteCallback((u,t,integrator) -> true, project!)
sdeprob = SDEProblem(f!, noise!, u0, tspan, params)
sol_sde = solve(sdeprob, EM(), dt=dt, saveat=dt, abstol=1e-8, reltol=1e-6, callback=proj_cb)

# === Plotting ===
using Plots, Measures, LaTeXStrings

plot_layout = @layout [a b ;c d ;e f]
p = plot(layout = plot_layout, size = (1200, 800), fontfamily = "Computer Modern", legend = true, leftmargin = 10mm, rightmargin = 5mm, bottommargin = 5mm)

# Plot each variable
plot!(p[1], sol.t, sol[1, :], label = "Xa ODE", xlims = (0, 40), ylims = (-0.1, 20), xlabel = "Time / h", ylabel = "Concentration / (g/L)", lw = 2)
plot!(p[1], sol_sde.t, sol_sde[1, :], label = "Xa SDE", linestyle = :dash, lw = 2)
# scatter!(p[1], df.time, df.Xa, label = "Measured")

plot!(p[2], sol.t, sol[2, :], label = "Xi ODE", xlims = (0, 40), ylims = (-0.2, 20), xlabel = "Time / h", ylabel = "Concentration / (g/L)", lw = 2)
plot!(p[2], sol_sde.t, sol_sde[2, :], label = "Xi SDE", linestyle = :dash, lw = 2)
# scatter!(p[2], df.time, df.Xi, label = "Measured")

plot!(p[3], sol.t, sol[3, :], label = "N ODE", xlims = (0, 40), xlabel = "Time / h", ylabel = "Concentration / (g/L)", ylims = (-0.01, 1.0), lw = 2)
plot!(p[3], sol_sde.t, sol_sde[3, :], label = "N SDE", linestyle = :dash, lw = 2)
# scatter!(p[3], df.time, df.N, label = "Measured")

plot!(p[4], sol.t, sol[4, :], label = "Suc ODE", xlims = (0, 40), ylims = (-0.5, 70), xlabel = "Time / h", ylabel = "Concentration / (g/L)", lw = 2)
plot!(p[4], sol_sde.t, sol_sde[4, :], label = "Suc SDE", linestyle = :dash, lw = 2)
# scatter!(p[4], df.time, df.S, label = "Measured")

plot!(p[5], sol.t, sol[5, :], label = "FruGlu ODE", xlims = (0, 40), ylims = (-0.5, 76), ylabel = "Concentration / (g/L)", xlabel = "Time / h", lw = 2)
plot!(p[5], sol_sde.t, sol_sde[5, :], label = "FruGlu SDE", linestyle = :dash, lw = 2)
# scatter!(p[5], df.time, df.FG, label = "Measured");

plot!(p[6], sol.t, sol[6, :], label = "Malic Acid ODE", xlims = (0, 40), ylims = (-0.3, 25), xlabel = "Time / h", ylabel = "Concentration / (g/L)", lw = 2)
plot!(p[6], sol_sde.t, sol_sde[6, :], label = "Malic Acid SDE", linestyle = :dash, lw = 2)
# scatter!(p[6], df.time, df.MA, label = "Measured");

# Display the plot
display(p)

# Save the plot
savefig(p, "Figures/kineticsMA_plot_3x2_odesde.pdf")