# NOTE: This code copy-pasted directly from Flux/layers/basic.jl

"""
    Dense(in::Integer, out::Integer, σ = identity)

Create a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(5))
2-element Array{Float32,1}:
  -0.16210233
   0.12311903```
"""
struct Dense{F,S,T}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(initW(out, in), initb(out), σ)
end

@functor Dense

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::Dense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Dense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

"""
    outdims(l::Dense, isize)

Calculate the output dimensions given the input dimensions, `isize`.

```julia
m = Dense(10, 5)
outdims(m, (5, 2)) == (5,)
outdims(m, (10,)) == (5,)
```
"""
outdims(l::Dense, isize) = (size(l.W)[1],)
