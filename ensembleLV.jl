using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations, Plots, Random, Distributions

function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end

function g(du, u, p, t)
    du[1] = p[3] * u[1]
    du[2] = p[4] * u[2]
end

p = [1.5, 1.0, 0.1, 0.1]
prob = SDEProblem(f, g, [1.0, 1.0], (0.0, 10.0), p)

# `p` is a global variable, referencing it would be type unstable.
# Using a let block defines a small local scope in which we can
# capture that local `p` which isn't redefined anywhere in that local scope.
# This allows it to be type stable.
prob_func = let p = p
    (prob, i, repeat) -> begin
        x = 0.3rand(2)
        remake(prob, p = vcat(p[1:2], [x[1], x[2]]))
    end
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, SRIW1(), trajectories = 10)
plot(sim, linealpha = 0.6, color = :blue, idxs = (0, 1), title = "Phase Space Plot");
plot!(sim, linealpha = 0.6, color = :red, idxs = (0, 2), title = "Phase Space Plot")

summ = EnsembleSummary(sim, 0:0.1:10)
plot(summ, fillalpha = 0.5)