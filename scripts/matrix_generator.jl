using JLD2
using LinearAlgebra
using Printf
using Random

rng = MersenneTwister(2025)
size_n = 10
size_m = 10
B = randn(rng, size_n, size_m)
cond_B = cond(B)

function save_matrix_with_info(B, cond_B)
    m, n = size(B)
    cond_str = @sprintf("%.2f", cond_B)

    base_dir = pwd()  
    folder = joinpath(base_dir, "data", "test_matrix", "$(m)x$(n)")
    mkpath(folder)

    filename = joinpath(folder, "$(m)x$(n)_cond$(cond_str).jld2")

    try
        @save filename B cond_B
        absfile = abspath(filename)
        sz = stat(absfile).size
        println("Matrix saved to: ", absfile)
        # println("   Condition number: ", cond_B)
        # println("   Size: ", size(B))
        # println("   File size: ", sz, " bytes")
    catch e
        @error "Error saving file" exception=e
    end
end

save_matrix_with_info(B, cond_B)