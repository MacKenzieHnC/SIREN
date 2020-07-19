using NNlib: conv, ∇conv_data, depthwiseconv, output_size, DenseConvDims
import Flux

# pad dims of x with dims of y until ndims(x) == ndims(y)
_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

_convtransoutdims(isize, ksize, ssize, dsize, pad) = (isize .- 1).*ssize .+ 1 .+ (ksize .- 1).*dsize .- (pad[1:2:end] .+ pad[2:2:end])

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)
"""
    Conv(size, in => out, σ = identity; init = glorot_uniform,
         stride = 1, pad = 0, dilation = 1)

Standard convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order (width, height, # channels, batch size).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

# Examples

Apply a `Conv` layer to a 1-channel input using a 2×2 window size, giving us a
16-channel output. Output is activated with ReLU.
```julia
size = (2,2)
in = 1
out = 16
Conv(size, in => out, relu)
```
"""
struct Conv{N,M,F,A,V}
  omega_0::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

function Conv(w::AbstractArray{T,N}, b::AbstractVector{T};
              omega_0 = 30, stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return Conv(omega_0, w, b, stride, pad, dilation)
end

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
     omega_0 = 30, is_first = false,  stride = 1, pad = 0, dilation = 1) where N =
  Conv(SIREN_init(omega_0, is_first, k..., ch...), zeros(ch[2]),
       omega_0=omega_0, stride = stride, pad = pad, dilation = dilation)

Flux.@functor Conv

function (c::Conv)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  omega_0, b = c.omega_0, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  sin.(omega_0 * (conv(x, c.weight, cdims) .+ b))
end

function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  print(io, ", omega_0=", l.omega_0)
  print(io, ")")
end

(a::Conv{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Conv{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))
