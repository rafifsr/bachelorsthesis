using Pkg
Pkg.activate(@__DIR__)
using Random
using Plots
using DifferentialEquations

# 1. Custom Euler-Maruyama with Clipped dBt
function euler_maruyama_clipped(f, g, x0, tspan, dt, C)
    t0, tf = tspan
    N = Int(round((tf - t0) / dt))
    t = range(t0, length=N+1, step=dt)
    x = zeros(length(x0), N+1)
    x[:, 1] = x0

    for i in 1:N
        ξ = randn(length(x0))
        dBt = sqrt(dt) .* ξ
        for j in 1:length(dBt)
            if abs(dBt[j]) > C * dt
                dBt[j] = sign(dBt[j]) * C * dt
            end
        end
        x[:, i+1] = x[:, i] + f(x[:, i], t[i]) * dt + g(x[:, i], t[i]) .* dBt
    end

    return t, x
end

# 2. GBM parameters
μ = 0.1
σ = 0.3
x0 = [1.0]
tspan = (0.0, 10.0)
dt = 0.01
C = 5.0

f(x, t) = μ * x
g(x, t) = σ * x

# 3. Run Euler-Maruyama
t_clipped, x_clipped = euler_maruyama_clipped(f, g, x0, tspan, dt, C)

# 4. Define SDE Problem for DifferentialEquations.jl
function drift!(du, u, p, t)
    du[1] = μ * u[1]
end

function diffusion!(du, u, p, t)
    du[1] = σ * u[1]
end

u0 = x0
prob = SDEProblem(drift!, diffusion!, u0, tspan)
sol = solve(prob, EM(), dt=dt)

# 5. Plot Comparison
plot(t_clipped, x_clipped[1, :], label="Euler-Maruyama (clipped)", lw=2)
plot!(sol.t, [u[1] for u in sol.u], label="DifferentialEquations.jl", lw=2, linestyle=:dash)
xlabel!("Time")
ylabel!("x(t)")
title!("Geometric Brownian Motion: Clipped EM vs. DE.jl")

