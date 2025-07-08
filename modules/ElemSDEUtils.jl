module ElemSDEutils
    using Pkg
    Pkg.activate(@__DIR__)
    using DifferentialEquations, Random, Distributions, Statistics

    export kinetics, simulate_paths, laguerre_design_matrix

    function kinetics(
        params::Dict, 
        T::Float64,
        u0::Dict, 
        noise::String = "noise",
        dt::Float64 = 0.01)

        tspan = (0.0, T)  # Time span for the simulation
        u_0 = [u0["X0"], u0["Y0"], u0["Z0"]]  # Initial conditions for X, Y, Z
        k1 = params["k1"]  # Rate constant for X
        k2 = params["k2"]  # Rate constant for Y
        σ = params["σ"]    # Noise scaling parameter
        K_sy = params["K_sy"]  # Saturation constant for Y
        p = (k1, k2, σ, K_sy)  # Parameters for the ODE/SDE

        # Define the drift and diffusion functions
        function drift!(du, u, p, t)
            X, Y, Z = u
            k1, k2 = p[1:2]
            du[1] = -k1 * X
            du[2] = k1 * X - k2 * Y
            du[3] = k2 * Y
        end

        function noise!(du, u, p, t)
            X, Y, Z = u
            σ, K_sy = p[3:4]
            du[1] = 0.0
            du[2] = σ * Y
            du[3] = 0.0
        end

        function monodnoise!(du, u, p, t)
            X, Y, Z = u
            σ, K_sy = p[3:4]
            du[1] = 0.0
            du[2] = σ * (Y / (K_sy + Y))
            du[3] = 0.0
        end

        # Project solution to stay ≥ 0
        function project!(integrator)
            integrator.u .= max.(integrator.u, 0.0)
        end
        proj_cb = DiscreteCallback((u,t,integrator) -> true, project!)

        # Create the ODE/SDE problem
        odeprob = ODEProblem(drift!, u_0, tspan, p)

        if noise == "noise"
            sdeprob = SDEProblem(drift!, noise!, u_0, tspan, p)
        elseif noise == "monod"
            sdeprob = SDEProblem(drift!, monodnoise!, u_0, tspan, p)
        else
            error("Invalid noise type. Choose 'noise' or 'monod'.")
        end

        # Solve the ODE/SDE problem
        odesol = solve(odeprob, Tsit5(), dt=dt, saveat=dt)
        sdesol = solve(sdeprob, EM(), dt=dt, saveat=dt, callback=proj_cb)

        return sdesol, odesol
    end # function kinetics

    # Run M simulations and collect results
    function simulate_paths(
        params::Dict, 
        T::Float64,
        u0::Dict, 
        M::Int,
        noise::String = "noise",
        dt::Float64 = 0.01)

        sderes = Vector{Any}(undef, M)
        for i in 1:M
            sderes[i] = kinetics(params, T, u0, noise, dt)[1]
        end
        oderes = kinetics(params, T, u0, noise, dt)[2]

        return sderes, oderes
    end # function simulate_paths

    # Laguerre basis functions
    function laguerre_design_matrix(
        y::Vector{Float64}, 
        d::Int)

        Φ = zeros(length(y), d + 1)
        for i in 1:length(y)
            Φ[i,1] = 1.0
            if d >= 1
                Φ[i,2] = 1 - y[i]
            end
            if d >= 2
                Φ[i,3] = 1 - 2*y[i] + 0.5*y[i]^2
            end
            if d >= 3
                Φ[i,4] = 1 - 3*y[i] + 1.5*y[i]^2 - (1/6)*y[i]^3
            end
        end
        return Φ
    end # function laguerre_design_matrix

end # module ElemSDEutils