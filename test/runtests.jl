using Revise
using SIREN
using Test

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
end


W = [1 2; 3 4]
x = [5; 6]
b = [7; 8]
ω_0 = 30

d = SIREN.Dense(W, b, ω_0)

sin.(ω_0 * W * x + b)

@testset "SIREN.jl" begin
    test_uniform()
    test_dense()
end
