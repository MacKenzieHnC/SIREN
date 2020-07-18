import Flux

function Dense(in::Integer, out::Integer;
                omega_0 = 30, is_first = false, initb = zeros)
  return Flux.Dense(SIREN_init(omega_0, is_first, out, in), initb(out), sin)
end
