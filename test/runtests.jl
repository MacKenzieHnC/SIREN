using SIREN
using Test

@testset "SIREN.jl" begin
    a = []

    for i in SIREN.uniform(100,100)
        if i < -1 || i > 1
            push!(a, i)
        end
    end
    @test length(a) == 0
end
