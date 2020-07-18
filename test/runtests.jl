using Revise
using SIREN
using Test
using Flux: params

w = SIREN_init(30, false, 5, 2)
d = SIREN.Dense(3, 2)

function test_uniform()
    a = []

    for i in SIREN.uniform(100,100)
        if i < -1 || i > 1
            push!(a, i)
        end
    end
    @test length(a) == 0
end

function test_dense()
    fan_in = 5
    fan_out = 3
    W = uniform(fan_out, fan_in)
    x = uniform(fan_in)
    b = uniform(fan_out)
    omega_0 = 30

    d = SIREN.Dense(W, b, omega_0)

    @test params(d)[1] == W
    @test params(d)[2] == b

    @test d(x) == sin.(omega_0 * W * x .+ b)
end

function test_conv()
    SIREN.Conv((3,2), 4=>5)
end

@testset "SIREN.jl" begin
    test_uniform()
    test_dense()
    test_conv()
end
