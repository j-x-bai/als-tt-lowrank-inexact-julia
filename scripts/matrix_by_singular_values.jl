using LinearAlgebra
using Random
using JLD2
using Printf


function rand_orthogonal(m::Int; rng=Random.default_rng())
    F = randn(rng, m, m)
    Q, R = qr!(F)
    return Matrix(Q)            
end

function matrix_with_singular_values(s::AbstractVector{<:Real},
                                     m::Integer, n::Integer; rng=Random.default_rng())
    # m rows, n columns
    r = min(m, n)
    @assert length(s) == r
    @assert all(x -> x > 0, s) # all singular values must be positive

    s_sorted = sort(collect(s); rev=true) # reverse=true for descending order

    U = rand_orthogonal(m; rng=rng)
    V = rand_orthogonal(n; rng=rng)
    Sigma = zeros(m, n)
    Sigma[1:r, 1:r] .= Diagonal(s_sorted)
    return U * Sigma * V'
    
end


function save_matrix_by_sv_with_info(B, cond_B, singular_values_B, sv_base_B)
    m, n = size(B)
    cond_str = @sprintf("%.2f", cond_B)

    base_dir = pwd()  
    folder = joinpath(base_dir, "data", "matrix_by_singular_values", "$(m)x$(n)")
    mkpath(folder)

    filename = joinpath(folder, "$(m)x$(n)_cond$(cond_str)_a=$(sv_base_B).jld2")

    try
        @save filename B cond_B singular_values_B sv_base_B
        sz = stat(filename).size
        println(" Matrix_by_sigular_values saved to: ", filename)
        # println("   Condition number: ", cond_B)
        # println("   Size: ", size(B))
        # println("   File size: ", sz, " bytes")
    catch e
        @error "Error saving file" exception=e
    end
end

rng = MersenneTwister(2025)
m, n = 100, 100                           # m rows, n columns
sv_bases = round.([i * 0.1 for i in 1:9]; digits=1)
for sv_base in sv_bases
    generated_singular_values = [sv_base^i for i in 1:100]           # desired singular values, length must be min(m,n), all positive     
    A = matrix_with_singular_values(generated_singular_values, m, n; rng=rng)
    cond_A = cond(A)
    save_matrix_by_sv_with_info(A, cond_A, generated_singular_values, sv_base)
end 