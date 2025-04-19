using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations
using Statistics

# Drift: logistic growth
function f(du, u, p, t)
    r = p[1]
    du[1] = r * u[1] * (1 - u[1])
end

# Diffusion: multiplicative noise
function g(du, u, p, t)
    σ = p[2]
    du[1] = σ * u[1]
end

u0 = [0.01]              # Initial biomass (normalized)
T = 100.0               # Final time
tspan = (0.0, T)      
p = (1.0, 0.2)           # r=1.0, σ=0.2  
t_det = 0.0:0.01:T

prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, SRIW1())
det_sol = u_0 * exp.(p[1] .* t_det) ./ (1 .+ u_0 * (exp.(p[1] * t_det) .- 1))

using Plots
using LaTeXStrings
plot(sol, label=L"\mathbb{E}[X_t]", xlabel="Time", ylabel="Expected Biomass", title="Stochastic Logistic Growth Model", fontfamily="Computer Modern")
plot!(t_det, det_sol, linestyle=:dash, label="Deterministic Solution", legend=:topright, fontfamily="Computer Modern")
xlims!(0, T)
ylims!(0, 2)