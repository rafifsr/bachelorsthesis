module SDEutils

    using Pkg
    Pkg.activate(@__DIR__)
    using DifferentialEquations, Random, Distributions, Statistics

    export kinetics, simulate_paths, laguerre_design_matrix

    """
    kinetics(params::Dict, T::Float64, u0::Dict; dt::Float64=0.01)

    Simulates the kinetics of a system using a stochastic differential equation (SDE) model.

    # Arguments
    - `params::Dict`: A dictionary containing the parameters for the kinetics. Expected keys:
        - `"μ_max"`: Maximum specific growth rate.
        - `"K_sx"`: Saturation constant for substrate.
        - `"Y_xs"`: Yield coefficient for biomass on substrate.
        - `"Y_ys"`: Yield coefficient for product on substrate.
        - `"m_s"`: Maintenance coefficient for substrate.
        - `"q_max_y"`: Maximum specific production rate of product.
        - `"K_sy"`: Saturation constant for product.
        - `"σs"`: Noise scaling parameter for substrate.
        - `"σx"`: Noise scaling parameter for biomass.
        - `"σy"`: Noise scaling parameter for product.
    - `T::Float64`: Total simulation time.
    - `u0::Dict`: Initial conditions as a dictionary with keys `"S0"`, `"X0"`, and `"Y0"`.
    - `dt::Float64`: Time step for the simulation (default is `0.01`).

    # Returns
    - `sol`: The solution object containing the simulated paths for substrate, biomass, and product concentrations.

    # Notes
    - The function uses a discrete callback to ensure that the solution remains non-negative.
    """

    function kinetics(
        params::Dict, # Parameters for the kinetics
        T::Float64, # Total time for the simulation
        u0::Dict, # Initial conditions
        dt::Float64 = 0.01) # Default time step
        
        # Biomass kinetics
        μ_max = params["μ_max"] 
        K_sx = params["K_sx"] 
        
        # Substrate kinetics
        Y_xs = params["Y_xs"]
        Y_ys = params["Y_ys"]
        m_s = params["m_s"]
        
        # Product kinetics
        q_max_y = params["q_max_y"]
        K_sy = params["K_sy"]

        # Noise scaling parameters
        σs = params["σs"]
        σx = params["σx"]
        σy = params["σy"]

        # Substrate noise parameters
        K_ss = K_sx/Y_xs + K_sy/Y_ys
        µ_s = μ_max/Y_xs + q_max_y/Y_ys

        # Pack the parameters into a tuple
        p = (μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, σs, σx, σy, K_ss, µ_s)

        # Initial conditions
        S0, X0, Y0 = u0["S0"], u0["X0"], u0["Y0"]
        u_0 = [S0, X0, Y0]

        # Drift term
        function drift!(du, u, p, t)
            S, X, Y = u

            μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy = p[1:7]
        
            μ_x = μ_max * (S / (K_sx + S))
            q_py = q_max_y * (S / (K_sy + S))
            
            # BioVT
            dS = - (1 / Y_xs) * μ_x * X - (1 / Y_ys) * q_py * X - m_s * X
            dX = μ_x * X
            dY = q_py * X
            
            du[1] = dS
            du[2] = dX
            du[3] = dY
        end

        # Diffusion term
        function diffusion!(du, u, p, t)
            S, X, Y = u
            μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, σs, σx, σy, K_ss, µ_s = p
            du[1] = σs * pdf(LogNormal(log(K_ss) + µ_s^2, µ_s), S) * X
            du[2] = σx * pdf(LogNormal(log(K_sx)+ μ_max^2, μ_max), S) * X
            du[3] = σy * pdf(LogNormal(log(K_sy)+ q_max_y^2, q_max_y), S) * X
        end
    
        # function diffusion!(du, u, p, t)
        #     S, X, Y = u
        #     μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, σs, σx, σy, K_ss, µ_s = p
        #     du[1] = σs * pdf(Normal(K_ss , µ_s), S) * X
        #     du[2] = σx * pdf(Normal(K_sx , μ_max), S) * X
        #     du[3] = σy * pdf(Normal(K_sy , q_max_y), S) * X
        # end
        

        # Project solution to stay ≥ 0
        function project!(integrator)
            integrator.u .= max.(integrator.u, 0.0)
        end
        proj_cb = DiscreteCallback((u,t,integrator) -> true, project!)

        # # Instantiate noise
        # ϵ = µ_s/K_ss # Bounded noise parameter
        # bn = BoundedNoise(ϵ, 3) # Create a bounded noise instance

        # Define the SDE problem
        prob = SDEProblem(drift!, diffusion!, u_0, (0.0, T), p)

        # Solve the SDE problem
        sol = solve(prob, EM(), dt = dt, saveat = dt, callback = proj_cb)
        return sol
    end # end the function kinetics

    """
    simulate_paths(params::Dict, T::Float64, u0::Dict, M::Int; dt::Float64=0.01)

    Simulates multiple paths of the system using the `kinetics` function.

    # Arguments
    - `params::Dict`: A dictionary containing the parameters for the simulation (see `kinetics` for details).
    - `T::Float64`: Total simulation time.
    - `u0::Dict`: Initial conditions as a dictionary with keys `"S0"`, `"X0"`, and `"Y0"`.
    - `M::Int`: Number of paths to simulate.
    - `dt::Float64`: Time step for the simulation (default is `0.01`).

    # Returns
    - `solutions::Vector{Any}`: A vector containing the solution objects for each simulated path.
    """

    function simulate_paths(
        params::Dict, # Parameters for the simulation
        T::Float64, # Total time for the simulation
        u0::Dict, # Initial conditions
        M::Int, # Number of paths to simulate
        dt::Float64 = 0.01) # Default time step

        solutions = Vector{Any}(undef, M)
        for i in 1:M
            solutions[i] = kinetics(params, T, u0, dt)
        end
        return solutions
    end # end the function simulate_paths

    """
    laguerre_design_matrix(y::Vector{Float64}, d::Int)

    Generates a design matrix using Laguerre basis functions.

    # Arguments
    - `y::Vector{Float64}`: Input vector for which the Laguerre basis functions are computed.
    - `d::Int`: Degree of the Laguerre basis functions.

    # Returns
    - `Φ::Matrix{Float64}`: A matrix where each row corresponds to the Laguerre basis functions evaluated at the corresponding element of `y`.

    # Notes
    - The Laguerre basis functions are computed up to degree `d`. If `d` is greater than 3, only the first four basis functions are implemented.
    """

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
    end # end the function laguerre_design_matrix

end # end the module SDEutils.jl