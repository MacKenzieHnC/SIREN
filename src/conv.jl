import Flux

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function Conv(w::AbstractArray{T,N}, b::AbstractVector{T};
              omega_0 = 30, is_first = false, stride = 1, pad = 0, dilation = 1) where {T,N}
    stride = expand(Val(N-2), stride)
    pad = expand(Val(2*(N-2)), pad)
    dilation = expand(Val(N-2), dilation)

    ## MY CODE
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
    ## /MY CODE

  return Flux.Conv(sin, w, b, stride, pad, dilation)
end

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
     omega_0 = 30, is_first = false, stride = 1, pad = 0, dilation = 1) where N =
  Conv(uniform(k..., ch...), zeros(ch[2]),
       stride = stride, pad = pad, dilation = dilation)

## ConvTranspose
function ConvTranspose(w::AbstractArray{T,N}, b::AbstractVector{T};
        omega_0 = 30, is_first = false, stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)

  ## MY CODE
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
  ## /MY CODE
  
  return Flux.ConvTranspose(sin, w, b, stride, pad, dilation)
end

ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
        omega_0 = 30, is_first = false, init = uniform, stride = 1, pad = 0, dilation = 1) where N =
    ConvTranspose(init(k..., reverse(ch)...), zeros(ch[2]),
        omega_0=omega_0, is_first=is_first, stride = stride, pad = pad, dilation = dilation)
