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
