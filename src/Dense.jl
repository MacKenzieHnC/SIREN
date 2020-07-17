import Flux

function Dense(in::Integer, out::Integer;
               omega_0 = 30, is_first = false)

    W = uniform(out, in)
    if is_first
        W ./= in
    else
        W .*= sqrt(6/in) / omega_0
    end

    return Flux.Dense(W, zeros(out), sin)
end
