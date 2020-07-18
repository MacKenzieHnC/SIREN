import Flux

## Conv Layer
Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
     omega_0 = 30, is_first = false, stride = 1, pad = 0, dilation = 1) where N =
  Flux.Conv(SIREN_init(omega_0, is_first, k..., ch...), zeros(ch[2]), sin,
       stride = stride, pad = pad, dilation = dilation)

## ConvTranspose
ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
        omega_0 = 30, is_first = false, stride = 1, pad = 0, dilation = 1) where N =
    Flux.ConvTranspose(SIREN_init(k..., reverse(ch)...), zeros(ch[2]), sin,
        stride = stride, pad = pad, dilation = dilation)
