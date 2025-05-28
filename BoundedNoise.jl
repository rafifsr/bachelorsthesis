import Base:copy

struct BoundedNoise{T}
    ϵ::T # |ΔBₜ| ≤ ϵ ⋅ Δt
    dim::Int
end

function copy(bn::BoundedNoise)
    return BoundedNoise(bn.ϵ, bn.dim)
end

function reset!(bn::BoundedNoise)
    return nothing
end

function (bn::BoundedNoise)(integrator)
    dt = integrator.dt
    dBt = sqrt(dt) * randn(bn.dim)
    max_jump = bn.ϵ * dt
    return clamp.(dBt, -max_jump, max_jump)
end