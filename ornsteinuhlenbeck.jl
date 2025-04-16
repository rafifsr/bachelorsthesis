using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations
using Plots

# Parameters
x_0 = 1.0
σ = 1.0
r = 1.0

# Drift term
function f!(dx, x, p, t)
    dx[1] = -r * x[1]
end

# Diffusion term
function g!(dx, x, p, t)
    dx[1] = σ
end

# Initial condition
x0 = [x_0]

# Time span
tspan = (0.0, 100.0)

# Problem definition
prob = SDEProblem(f!, g!, x0, tspan)

# Solve the SDE
sol = solve(prob, EM(), dt=0.01)

# Expected path
times = sol.t
expected = x0[1] * exp.(-r .* times) 

plot(times, expected, label="Expected path", lw=2, linestyle=:dash)
plot!(sol, label="OU Path", xlabel="Time", ylabel="X(t)", title="Ornstein-Uhlenbeck Process", 
     legend=:topright, grid=true)
xlims!(0, maximum(times))
ylims!(-3, 3)

# Standard deviation
std_dev = σ * sqrt.(1 .- exp.(-2 .* r .* times) ./ (2 * r))

# Plot plus and minus standard deviation
plot!(times, expected .+ std_dev, label="Mean + Std Dev", lw=1, linestyle=:dot)
plot!(times, expected .- std_dev, label="Mean - Std Dev", lw=1, linestyle=:dot)