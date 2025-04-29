using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, LinearAlgebra, Statistics, Plots

# Drift term
function monod(du, u, p, t)
    S, X, P = max.(u, 0.0)           # u[1] = substrate S, u[2] = biomass X, u[3] = product P
    μ_max, K_s, Y_s, q_max, K_sp, Y_p = p      # unpack parameters

    μ = μ_max * S / (K_s + S + 1e-6)  # Monod growth rate
    q = q_max * S / (K_sp + S + 1e-6)  # Monod product formation rate
    dX = μ * X
    dS = - (1 / Y_s) * μ * X - (1 / Y_p) * q * X  # Monod substrate consumption rate
    dP = q * X

    du[1] = dS
    du[2] = dX
    du[3] = dP
end

# Diffusion term
function noise(du, u, p, t)
    S, X, P = max.(u, 0.0)  # Ensure non-negative concentrations
    σ1 = p[7]
    σ2 = p[8]
    du[1] = σ1*S
    du[2] = σ2*X
    du[3] = 0
end

# Initial conditions and parameters
X₀ = 0.5  # g/L
S₀ = 10.0 # g/L
P₀ = 0.0  # g/L
u₀ = [S₀, X₀, P₀]  # Initial concentrations of X, S, and P

# Parameters: μ_max, K_s, Y_s, q_max, K_sp, Y_p, σ1, σ2
p = (0.3, 2, 0.5, 0.5, 0.5, 0.5, 0.05, 0.05)
tspan = (0.0, 30.0)  # Time span

# Problem definition
prob = SDEProblem(monod, noise, u₀, tspan, p)

# Solve the SDE problem
# Callback to enforce positivity
positive_cb = ContinuousCallback(
    (u, t, integrator) -> minimum(u),
    (integrator) -> integrator.u .= max.(integrator.u, 0.0)
)
sol = solve(prob, SOSRI(), callback=positive_cb, dt=0.01, saveat=0.1)

# Plot the solution
plot(sol, label=["Substrate (S)" "Biomass (X)" "Product (P)"], xlabel="Time (h)", ylabel="Concentration (g/L)", lw=2, legend=:topright, 
     title="Monod Kinetics", fontfamily="Computer Modern", ylims=(-1, 10), xlims=(0, 30), grid=true)