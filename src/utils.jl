"""
    uniform(dims...)

Return an `Array` of size `dims` containing random variables taken from a uniform
distribution in the interval ``[-1, 1]``.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> uniform(2, 3)
2×3 Array{Float32,2}:
 0.548721  -0.524116  -0.743921
 0.822376   0.735768   0.0525029
```
"""
uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) * 2

"""
    SIREN_init(SIREN_init(omega_0::Real, is_first::Bool, dims::Integer...)

Return an `Array` of size `dims` containing random variables.
The distribution is supposed to be an implementation of the SIREN initialization
from the original paper (Sec 3.2, Appdx 1, Appdx 1.5). My explanation will probably
be wrong so check out their explanation.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> SIREN_init(30,false,3,2)
3×2 Array{Float64,2}:
  0.0305138  -0.0304217
  0.0386888  -0.0208474
 -0.031625   -0.0279565
```
"""
function SIREN_init(omega_0::Real, is_first::Bool, dims::Integer...)
    w = uniform(dims)

    fan_in = 1
    s = size(w)
    for i in 1:(length(s)-1)
      fan_in *= s[i]
    end

    w .*= sqrt(6/fan_in)

    if !is_first
      w ./= omega_0
    end

    return w
end
