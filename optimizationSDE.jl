using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations
using Plots
using Statistics
using LaTeXStrings

# Parameters
r = 1.0      # intrinsic growth rate
σ = 0.2      # noise intensity
x_0 = 0.01     # initial biomass
T = 100.0     # final time

# Drift and diffusion functions
function f!(dx, x, p, t)
    dx[1] = r * x[1] * (1 - x[1])
end

function g!(dx, x, p, t)
    dx[1] = σ * x[1]
end

# Initial condition
x0 = [x_0]

# Time span
tspan = (0.0, T)
t_det = 0.0:0.01:T
# Problem definition
prob = SDEProblem(f!, g!, x0, tspan)

# Solve the SDE
sol = solve(prob, SRIW1(), dt=0.01)
plot(sol, label="Biomass X(t)", xlabel="Time", ylabel="X(t)", title="Stochastic Logistic Growth Model")

# Deterministic solution for comparison
det_sol = x_0 * exp.(r * t_det) ./ (1 .+ x_0 * (exp.(r * t_det) .- 1))

# Monte Carlo simulation
N = 1000
dt = 0.01
tsteps = 0.0:dt:T
X = zeros(length(tsteps), N)

for i in 1:N
    sol = solve(prob, EM(), dt=dt, saveat=tsteps)
    X[:, i] .= sol[1, :]
end

# Expected biomass at each time point
mean_X = mean(X, dims=2)

# Find time of max expected biomass
max_idx = argmax(mean_X)
optimal_time = tsteps[max_idx]

println("🔍 Estimated optimal stopping time: $optimal_time")
plot(tsteps, mean_X, label=L"\mathbb{E}[X_t]", xlabel="Time", ylabel="Expected Biomass", title="Expected Biomass over Time", fontfamily="Computer Modern")
plot!(tsteps, det_sol, label="Deterministic Solution", linestyle=:dash)
vline!([optimal_time], label="Optimal Stop", linestyle=:dash)
xlims!(0, T)
ylims!(0, 1)