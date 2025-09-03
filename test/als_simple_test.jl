using IterativeSolvers
using LinearAlgebra
using JLD2
using DataFrames, CSV, PrettyTables

num_threads = Threads.nthreads()
println("Number of threads: ", num_threads)

function als_2d(B::AbstractMatrix, r::Number;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=100, outer_tol::Float64=1e-8,
                # inner_method::Symbol=:cg, 
                inner_maxiters::Int=100, inner_tol::Float64=1e-6)
    n, m = size(B)
    
    if X1 === nothing
        X1 = randn(n, r)
    end

    if X2 === nothing
        X2 = randn(m, r)
    end

    error = zeros(outer_maxiters)
    error[1] = norm(X1 * X2' - B, 2)
    Y1 = zeros(n, r)
    Y2 = zeros(m, r)
    converged_als = false
  
    for s in 1:outer_maxiters
        # Y1 = cg(X2' * X2, X2' * B; abstol=inner_tol, maxiter=inner_maxiters)
        # CG : right-hand side cannot be a matrix
        # apply CG to each column of the right-hand side matrix
        # println("s=", s)

        # Threads.@threads for j in 1:n
        #     Y1[j, :] = cg(X2' * X2, (X2' * B')[:, j]; abstol=inner_tol, maxiter=inner_maxiters)
        # end
        # X1 = Y1

        # Threads.@threads for j in 1:m
        #     Y2[j, :] = cg(X1' * X1, (X1' * B)[:, j]; abstol=inner_tol, maxiter=inner_maxiters)
        # end
        # X2 = Y2

        blocksize1 = ceil(Int, n / num_threads)
        Threads.@threads for core in 1:num_threads
            for j = (core-1)*blocksize1 +1 : min(core*blocksize1, n)
                Y1[j, :] = Y1[j, :] = cg(X2' * X2, (X2' * B')[:, j]; abstol=inner_tol, maxiter=inner_maxiters)
            end
        end
        X1 = Y1

        blocksize2 = ceil(Int, m / num_threads)
        Threads.@threads for core in 1:num_threads
            for j = (core-1)*blocksize2 +1 : min(core*blocksize2, m)
                Y2[j, :] = Y2[j, :] = cg(X1' * X1, (X1' * B)[:, j]; abstol=inner_tol, maxiter=inner_maxiters)
            end
        end
        X2 = Y2

        error[s] = norm(X1 * X2' - B, 2)
        if error[s] <= outer_tol
            println("Converged at outer iteration ", s)
            converged_als = true
            break
        end
    end

    B_als = X1 * X2'
    return B_als, error, converged_als
end

function svd_truncated(B::AbstractMatrix, rank::Number)

    B_svd = svd(B, full=false)
    U, S, V = Matrix(B_svd.U), B_svd.S, Matrix(B_svd.V)
    # println("size of S: ", size(S))
    # println("largest singular value: ", S[1])
    # println("second largest singular value: ", S[2])
    # println("smallest singular value: ", S[end])
    # println("size of USV': ", size(U * Diagonal(S) * V'))

    return U[:, 1:rank] * Diagonal(S[1:rank]) * V[:, 1:rank]', S
end


# @load "matrix/10x10/10x10_cond9.24.jld2" B cond_B
# size_n, size_m = size(B)
# # rank = 1
# rank = 5
# condition_number = cond_B
# println("condition number: ", condition_number)

# inner_maxiters = 100
# inner_tol = 1e-8
# outer_maxiters = 50
# outer_tol = 1e-8

# als_time = @elapsed B_als, als_error_his, converged_als = als_2d(B, rank; inner_maxiters=inner_maxiters, inner_tol=inner_tol, outer_maxiters=outer_maxiters, outer_tol=outer_tol)
# svdt_time = @elapsed B_svdt, singular_values = svd_truncated(B, rank)
# # println("Rate of convergence in Power method: ", singular_values[2] / singular_values[1])

# # println("1st als_error_his: ", als_error_his[1])
# # println("5nd als_error_his: ", als_error_his[5])
# # println("10th als_error_his: ", als_error_his[10])
# # println("20th als_error_his: ", als_error_his[20])
# # println("50th als_error_his: ", als_error_his[50])
# # println("70th als_error_his: ", als_error_his[70])
# # println("100th als_error_his: ", als_error_his[100])    

# als_error = als_error_his[end]
# println("Final error from ALS: ", als_error)
# svdt_error = norm(B_svdt - B, 2)
# println("Error of truncated SVD: ", svdt_error)
# als_svdt_error = norm(B_als - B_svdt, 2)
# println("Error X1 * X2' - B_svdt between ALS and truncated SVD: ", als_svdt_error)
# # error4 = norm(X2 * X1' - B, 2)
# # println("Error X2' * X1' - B between ALS and truncated SVD: ", error4)  
# println("ALS Converged ? ", converged_als)

# isdir("results_test") || mkpath("results_test")
# CSV.write("results_test/results.csv", results)

# if isfile("results_test/results.csv")
#     results = CSV.read("results_test/results.csv", DataFrame)
# else
#     results = DataFrame(
#         num_threads = Int[],
#         size_n = Int[],
#         size_m = Int[],
#         condition_number = Float64[],
#         rank = Int[],
#         outer_tol = Float64[],
#         outer_maxiters = Int[],
#         inner_tol = Float64[],
#         inner_maxiters = Int[],
#         als_error = Float64[],
#         svdt_error = Float64[],
#         als_svdt_error = Float64[],
#         converged_als = Bool[],
#         als_time = Float64[],
#         svdt_time = Float64[]
#     )
# end

# push!(results, (
#     threads = num_threads,      
#     n = n,                
#     m = m,                
#     cond_num = cond_B,           
#     rank = rank,             
#     outer_tol = outer_tol,        
#     outer_maxiters = outer_maxiters,   
#     inner_tol = inner_tol,        
#     inner_maxiters = inner_maxiters,   
#     als_error = als_error,        
#     svdt_error = svdt_error,       
#     als_svdt_error = als_svdt_error,  
#     converged_als = converged_als,
#     als_time = als_time,         
#     svdt_time = svdt_time         
# ))



# using Base.Threads
# nthreads()

# Threads.nthreads()

# export PATH="/Applications/Julia-1.11.app/Contents/Resources/julia/bin:$PATH"
# source ~/.zshrc
# julia --threads 4 als_simple_test.jl