using Random
"""
    uniform(dims...)

Return an `Array` of size `dims` containing random variables taken from a uniform
distribution in the interval ``[-1, 1]``.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> SIREN.uniform(2, 3)
2×3 Array{Float32,2}:
 0.548721  -0.524116  -0.743921
 0.822376   0.735768   0.0525029
```
"""
uniform(dims...) = (rand(Float64, dims...) .- 0.5f0) * 2
function SIREN_init(omega_0::Real, is_first::Bool, dims...)
    w = uniform(dims)

    fan_in = 1
    s = size(w)
    for i in 1:(length(s)-1)
      fan_in *= s[i]
    end
    if is_first
      w ./= fan_in
    else
      w .*= sqrt(6/fan_in) / omega_0
    end

    return w
end
