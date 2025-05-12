using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, LinearAlgebra, Statistics, Plots

# Drift term
function monod(du, u, p, t)
    S, X, P, Z= u          # u[1] = substrate S, u[2] = biomass X, u[3] = product P
    μ_max, K_s, Y_s, q_max, K_sp, Y_p, k_d, k_z = p      # unpack parameters

    μ = μ_max * S / (K_s + S)  # Monod growth rate
    q = q_max * S / (K_sp + S)  # Monod product formation rate
    dX = μ * X - k_z * P * X - k_d * X # Growth rate with decay
    dS = - (1 / Y_s) * μ * X - (1 / Y_p) * q * X  # Monod substrate consumption rate
    dP = q * X - k_z * P * X
    dZ = k_z * P * X

    du[1] = dS
    du[2] = dX
    du[3] = dP
    du[4] = dZ
end

# Diffusion term
function noise(du, u, p, t)
    S, X, P, Z = u 
    σ1 = p[end-1]
    σ2 = p[end]

    du[1] = -(σ1+σ2)*X*S
    du[2] = σ1*X*S
    du[3] = σ2*X*S
    du[4] = σ2*X*P
end

# Initial conditions and parameters
X₀ = 1.0  # g/L
S₀ = 5.0 # g/L
P₀ = 0.0  # g/L
Z₀ = 0.0  # g/L
u₀ = [S₀, X₀, P₀, Z₀]  # Initial concentrations of S, X, and P

# Parameters: μ_max, K_s, Y_s, q_max, K_sp, Y_p, k_d, k_z, σ1, σ2
p = (0.33, 1.7, 0.08, 0.5, 0.5, 0.5, 0.45, 0.5, 0.05, 0.02)
tspan = (0.0, 30.0)  # Time span

# Problem definition
prob = SDEProblem(monod, noise, u₀, tspan, p)

# Solve the SDE problem
sol = solve(prob, EM(), dt=0.01, saveat=0.1)

# Plot the solution
plot(sol, label=["Substrate (S)" "Biomass (X)" "Product (P)" "Byproduct (Z)"], xlabel="Time (h)", ylabel="Concentration (g/L)", lw=2, legend=:topright, 
     title="Monod Kinetics", fontfamily="Computer Modern", ylims=(0, 5), xlims=(0, 10), grid=true)