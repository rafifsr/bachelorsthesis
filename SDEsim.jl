module SDEsim
using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations

# ------------------------------------------------------------------------
# simulate_paths — vectorised wrapper around DifferentialEquations.jl
# ------------------------------------------------------------------------
"""
    simulate_paths(f, g, x0;
                   T, N, M,
                   rng = Random.default_rng(),
                   alg = EM(),
                   kwargs...)

Simulate `M` Monte-Carlo paths of the scalar SDE

``dxₜ = f(xₜ, t) dt + g(xₜ, t) dWₜ``

from `t = 0` to `T`, starting at `x0`.

Arguments
---------
* `f(x, t)`         — drift function  (scalar → scalar)
* `g(x, t)`         — diffusion term (scalar → scalar)
* `x0`              — initial state (number)
* `T`               — final time
* `N`               — *nominal* number of time steps (used to choose `dt`)
* `M`               — number of trajectories (paths)
* `alg`             — SDE solver (default: Euler–Maruyama `EM()`)

Keyword `kwargs...` are forwarded to `solve`, e.g. `abstol`, `reltol`,
`dt = T/N`, `saveat = Δt`, etc.

Returns
-------
A matrix of size `(N+1, M)` whose columns are the simulated paths.
"""
function simulate_paths(f, g, x0;
                        T::Real,
                        N::Integer,
                        M::Integer,
                        alg = EM(),
                        kwargs...)

    # ---- 1 · wrap scalar drift/diffusion into in-place forms for DiffEq ----
    drift!(du, u, p, t)     = (du[1] = f(u[1], t))
    diffusion!(du, u, p, t) = (du[1] = g(u[1], t))

    # ---- 2 · base SDE problem (single trajectory) -------------------------
    prob  = SDEProblem(drift!, diffusion!, [x0], (0.0, T); noise_rate_prototype=zeros(1))

    # ---- 3 · ensemble problem for M independent trajectories --------------
    ensemble_prob = EnsembleProblem(prob)

    Δt      = T / N
    sol = solve(ensemble_prob, alg;
                trajectories = M,
                dt           = Δt,      # fixed step (override via kwargs)
                saveat       = 0:Δt:T,  # aligns all trajectories on common grid
                kwargs...)              # user overrides

    # ---- 4 · pack results into (N+1)×M matrix -----------------------------
    # Each `sol[i]` is a `Solution` object; its values are vectors of length 1
    paths = hcat((reduce(vcat, s.u) for s in sol)...)
    return paths                      # rows = time steps, cols = trajectories
end

export simulate_paths
end # module
