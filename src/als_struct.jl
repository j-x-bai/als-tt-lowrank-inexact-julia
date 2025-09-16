using IterativeSolvers
using LinearAlgebra
# using BenchmarkTools
using Base.Threads
using JLD2
using Plots

# X1: n x r
# X2: m x r
# B: n x m
# B ≈ X1 * X2'

struct ALSResult 
    outer_iters::Int
    inner_iters::Vector{Int}
    total_inner_iters::Int
    als_error::Vector{Float64}
    converged_als::Bool
    time_cg::Vector{Float64}
    total_time_cg::Float64
    outer_tol::Float64
    inner_tol::Float64
end

function als_2d(B::AbstractMatrix, rank::Number, B_svd_truncated ::AbstractMatrix;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=5000, 
                outer_tol::Float64=1e-12,
                # inner_method::Symbol=:cg, 
                inner_maxiters::Int=5000, 
                inner_tol::Float64=1e-12)
    n, m = size(B)

    # Initialize starting points
    # In fact, only X2 is needed to initialize, since X1 is updated first.
    if X1 === nothing
        X1 = randn(n, rank)
    end
    if X2 === nothing
        X2 = randn(m, rank)
    end

    # # Precompute truncated SVD of B for error computation
    # B_svd = svd(B, full=false)
    # U, S, V = Matrix(B_svd.U), B_svd.S, Matrix(B_svd.V)
    # B_svd_truncated = U[:, 1:rank] * Diagonal(S[1:rank]) * V[:, 1:rank]'
    # svd_error = norm(B - B_svd_truncated, 2)
    
    # Initialize loop variables
    # Outer loop
    outer_iters = 0 # The first computation of initial X1 and X2 is not included
    als_error = zeros(Float64, outer_maxiters) # The error between ALS and truncated svd
    converged_als = false
    inner_iters = zeros(Int, outer_maxiters) # Total number of inner iterations at each outer step

    # Inner loop (Conjugate Gradient)
    Y1 = zeros(n, rank) # Temporary storage for updated X1
    Y2 = zeros(m, rank) # Temporary storage for updated X2
    # To store convergence history of each CG solve
    # How to initialize a vector we don't know the length of?
    ch1_j = nothing 
    ch2_j = nothing

    A1_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X1
    RHS1_cg = zeros(rank, n)  # Right-hand side for CG in updating X1
    A2_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X2
    RHS2_cg = zeros(rank, m)  # Right-hand side for CG in updating X2

    # Record time used for each sweep
    time_cg = zeros(Float64, outer_maxiters)

    # Initialize parallel processing
    num_threads = Threads.nthreads()
    blocksize1 = ceil(Int, n / num_threads)
    blocksize2 = ceil(Int, m / num_threads)

    iters_per_thread = zeros(Int, num_threads)
    time_per_thread  = zeros(Float64, num_threads)

    # Avoid the first-time overhead in timing
    Threads.@threads for core in 1:num_threads
        nothing
    end

    for s in 1:outer_maxiters

        # Parallelize the inner loop
        # Each thread computes several columns of Y1 (or Y2)

        # A1_cg .= X2' * X2
        # RHS1_cg .= X2' * B'
        mul!(A1_cg, X2', X2)     
        mul!(RHS1_cg, X2', B')   
        Threads.@threads for core in 1:num_threads
                    local_cg_s1_iters = 0
                    local_cg_j1_iters = 0
                    local_time1_j = 0.00
                    tid = threadid()
       
            for j = (core-1)*blocksize1 +1 : min(core*blocksize1, n)
                local_time1_j = @elapsed begin 
                    Y1[j, :], ch1_j = cg(A1_cg, @views RHS1_cg[:, j]; abstol=inner_tol, maxiter=inner_maxiters, log=true)
                end
                # local_cg_j1_iters = length(ch1_j.data[:resnorm])
                local_cg_j1_iters = ch1_j.iters
                local_cg_s1_iters += local_cg_j1_iters
                local_time1_j += local_time1_j
            end

            iters_per_thread[tid] = local_cg_s1_iters
            time_per_thread[tid] = local_time1_j
        end
        X1 .= Y1

        # A2_cg .= X1' * X1
        # RHS2_cg .= X1' * B
        mul!(A2_cg, X1', X1)
        mul!(RHS2_cg, X1', B)
        Threads.@threads for core in 1:num_threads
                    local_cg_s2_iters = 0
                    local_cg_j2_iters = 0
                    local_time2_j = 0.00
                    tid = threadid()
            for j = (core-1)*blocksize2 +1 : min(core*blocksize2, m)
                local_time2_j = @elapsed begin 
                    Y2[j, :], ch2_j = cg(A2_cg, @views RHS2_cg[:, j]; abstol=inner_tol, maxiter=inner_maxiters, log=true)
                end
                # local_cg_j2_iters = length(ch2_j.data[:resnorm])
                local_cg_j2_iters = ch2_j.iters
                local_cg_s2_iters += local_cg_j2_iters
                local_time2_j += local_time2_j
            end

            iters_per_thread[tid] += local_cg_s2_iters
            time_per_thread[tid] += local_time2_j
        end
        X2 .= Y2

        outer_iters += 1
        inner_iters[s] = sum(iters_per_thread)
        als_error[s] = norm(X1 * X2' - B_svd_truncated, 2)
        time_cg[s] = sum(time_per_thread)

        # fill!(iters_per_thread, 0)
        # fill!(time_per_thread, 0.0000)  

        if als_error[s] <= outer_tol
            println("The matrix ", n, "x", m, " with condition number ", cond(B), " and outer tolerance ", outer_tol, " and inner tolerance ", inner_tol, " converged at outer iteration: ", s)
            converged_als = true
            break
        end
    end
    # B_als = X1 * X2'
    resize!(als_error, outer_iters)
    resize!(inner_iters, outer_iters)
    resize!(time_cg, outer_iters)
    total_time_cg = sum(time_cg) # Total time used in all CG solves
    total_inner_iters = sum(inner_iters)

    # return B_svd_truncated, B_als, outer_iters, inner_iters, svd_error, error, converged_als, time_cg, total_time_cg
    result = ALSResult(outer_iters, inner_iters, total_inner_iters, als_error, converged_als, time_cg, total_time_cg, outer_tol, inner_tol)
    return result
    # return outer_iters, inner_iters, svd_error, als_error, converged_als, time_cg
end

function als_2d_qr(B::AbstractMatrix, rank::Number, B_svd_truncated ::AbstractMatrix;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=10000, 
                outer_tol::Float64=1e-12,
                # inner_method::Symbol=:cg, 
                inner_maxiters::Int=10000, 
                inner_tol::Float64=1e-12)
    n, m = size(B)

    # Initialize starting points
        # In fact, only X2 is needed to initialize, since X1 is updated first.
        if X1 === nothing
            F1 = qr(randn(n, rank))
            X1 = Matrix(F1.Q)#[:, 1:rank]
        else
            F1 = qr(X1)
            X1 = Matrix(F1.Q)#[:, 1:rank]
        end

        if X2 === nothing
            F2 = qr(randn(m, rank))
            X2 = Matrix(F2.Q)#[:, 1:rank]
        else
            F2 = qr(X2)
            X2 = Matrix(F2.Q)#[:, 1:rank]
        end

    # # Precompute truncated SVD of B for error computation
    # B_svd = svd(B, full=false)
    # U, S, V = Matrix(B_svd.U), B_svd.S, Matrix(B_svd.V)
    # B_svd_truncated = U[:, 1:rank] * Diagonal(S[1:rank]) * V[:, 1:rank]'
    # svd_error = norm(B - B_svd_truncated, 2)
    
    # Initialize loop variables
    # Outer loop
    outer_iters = 0 # The first computation of initial X1 and X2 is not included
    als_error = zeros(Float64, outer_maxiters) # The error between ALS and truncated svd
    converged_als = false
    inner_iters = zeros(Int, outer_maxiters) # Total number of inner iterations at each outer step

    # Inner loop (Conjugate Gradient)
    Y1 = zeros(n, rank) # Temporary storage for updated X1
    Y2 = zeros(m, rank) # Temporary storage for updated X2
    # To store convergence history of each CG solve
    # How to initialize a vector we don't know the length of?
    ch1_j = nothing 
    ch2_j = nothing

    A1_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X1
    RHS1_cg = zeros(rank, n)  # Right-hand side for CG in updating X1
    A2_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X2
    RHS2_cg = zeros(rank, m)  # Right-hand side for CG in updating X2

    # Record time used for each sweep
    time_cg = zeros(Float64, outer_maxiters)

    # Initialize parallel processing
    num_threads = Threads.nthreads()
    blocksize1 = ceil(Int, n / num_threads)
    blocksize2 = ceil(Int, m / num_threads)

    iters_per_thread = zeros(Int, num_threads)
    time_per_thread  = zeros(Float64, num_threads)

    # Avoid the first-time overhead in timing
    Threads.@threads for core in 1:num_threads
        nothing
    end

    for s in 1:outer_maxiters

        # Parallelize the inner loop
        # Each thread computes several columns of Y1 (or Y2)

        # A1_cg .= X2' * X2
        # RHS1_cg .= X2' * B'
        mul!(A1_cg, X2', X2)     
        mul!(RHS1_cg, X2', B')   
        Threads.@threads for core in 1:num_threads
                    local_cg_s1_iters = 0
                    local_cg_j1_iters = 0
                    local_time1_j = 0.00
                    tid = threadid()
       
            for j = (core-1)*blocksize1 +1 : min(core*blocksize1, n)
                local_time1_j = @elapsed begin 
                    Y1[j, :], ch1_j = cg(A1_cg, @views RHS1_cg[:, j]; abstol=inner_tol, maxiter=inner_maxiters, log=true)
                end
                # local_cg_j1_iters = length(ch1_j.data[:resnorm])
                local_cg_j1_iters = ch1_j.iters
                local_cg_s1_iters += local_cg_j1_iters
                local_time1_j += local_time1_j
            end

            iters_per_thread[tid] = local_cg_s1_iters
            time_per_thread[tid] = local_time1_j
        end
    # Orthonormalize X1 and compensate X2 to keep X1*X2' unchanged
    X1 .= Matrix(qr(Y1).Q)
    # F1 = qr(Y1)
    # Q1 = Matrix(F1.Q) #[:, 1:rank]
    # R1 = Martrix(F1.R) #[1:rank, 1:rank]
    # X1 .= Q1
    # X2 .= X2 * R1'

        # A2_cg .= X1' * X1
        # RHS2_cg .= X1' * B
        mul!(A2_cg, X1', X1)
        mul!(RHS2_cg, X1', B)
        Threads.@threads for core in 1:num_threads
                    local_cg_s2_iters = 0
                    local_cg_j2_iters = 0
                    local_time2_j = 0.00
                    tid = threadid()
            for j = (core-1)*blocksize2 +1 : min(core*blocksize2, m)
                local_time2_j = @elapsed begin 
                    Y2[j, :], ch2_j = cg(A2_cg, @views RHS2_cg[:, j]; abstol=inner_tol, maxiter=inner_maxiters, log=true)
                end
                # local_cg_j2_iters = length(ch2_j.data[:resnorm])
                local_cg_j2_iters = ch2_j.iters
                local_cg_s2_iters += local_cg_j2_iters
                local_time2_j += local_time2_j
            end

            iters_per_thread[tid] += local_cg_s2_iters
            time_per_thread[tid] += local_time2_j
        end
    # Orthonormalize X2 and compensate X1
    # F2 = qr(Y2)
    # Q2 = Matrix(F2.Q) #[:, 1:rank]
    # R2 = Matrix(F2.R) #[1:rank, 1:rank]
    # X2 .= Q2
    # X1 .= X1 * R2'
    X2 .= Matrix(qr(Y2).Q)
    X1 .= X1 * (Matrix(qr(Y2).R'))

        outer_iters += 1
        inner_iters[s] = sum(iters_per_thread)
        als_error[s] = norm(X1 * X2' - B_svd_truncated, 2)
        time_cg[s] = sum(time_per_thread)

        # fill!(iters_per_thread, 0)
        # fill!(time_per_thread, 0.0000)  

        if als_error[s] <= outer_tol
            println("The matrix ", n, "x", m, " with condition number ", cond(B), " and outer tolerance ", outer_tol, " and inner tolerance ", inner_tol, " converged at outer iteration: ", s)
            converged_als = true

            break
        end
    end
    # B_als = X1 * X2'
    resize!(als_error, outer_iters)
    resize!(inner_iters, outer_iters)
    resize!(time_cg, outer_iters)
    total_time_cg = sum(time_cg) # Total time used in all CG solves
    total_inner_iters = sum(inner_iters)

    # return B_svd_truncated, B_als, outer_iters, inner_iters, svd_error, error, converged_als, time_cg, total_time_cg
    result = ALSResult(outer_iters, inner_iters, total_inner_iters, als_error, converged_als, time_cg, total_time_cg, outer_tol, inner_tol)
    return result
    # return outer_iters, inner_iters, svd_error, als_error, converged_als, time_cg
end












    # Precompute truncated SVD of B for error computation
    # B_svd = svd(B, full=false)
    # U, S, V = Matrix(B_svd.U), B_svd.S, Matrix(B_svd.V)
    # B_svd_truncated = U[:, 1:rank] * Diagonal(S[1:rank]) * V[:, 1:rank]'
    # svd_error = norm(B - B_svd_truncated, 2)


function svd_truncated(B::AbstractMatrix, rank::Number)

    B_svd = svd(B, full=false)
    U, S, V = Matrix(B_svd.U), B_svd.S, Matrix(B_svd.V)
    B_svd_truncated = U[:, 1:rank] * Diagonal(S[1:rank]) * V[:, 1:rank]'
    svd_error = norm(B - B_svd_truncated, 2)

    s = Vector(B_svd.S)
    ratios = [s[i+1]/s[i] for i in range(1, length(s)-1)]

    return B_svd_truncated, svd_error, ratios, s
end

