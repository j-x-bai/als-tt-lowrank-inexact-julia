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
    rank::Int
    outer_iters::Int
    inner_iters::Vector{Int}
    total_inner_iters::Int
    als_error::Vector{Float64}
    converged_als::Bool
    time_cg::Vector{Float64}
    total_time_cg::Float64
    outer_tol::Float64
    inner_abstol::Float64
    inner_reltol::Float64
    cg_last_resnorm_1::Vector{Float64}
    cg_last_resnorm_2::Vector{Float64}
    X2tX2_fnorm::Float64
    bound_ratio::Vector{Float64}
    max_angle::Vector{Float64} # radians
    upper_error_bound::Vector{Float64}
    last_theta_max::Float64
    last_upper_error_bound::Float64
    last_theor_ratio::Float64
end


function als_2d_qr(B::AbstractMatrix, rank::Int, 
                cond_B::Float64,
                square_sigma_ratio::Float64,
                B_svd_truncated ::AbstractMatrix,
                SigmaV::AbstractMatrix;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=10000, 
                outer_tol::Float64=1e-12,
                # inner_method::Symbol=:cg, 
                inner_maxiters::Int=10000, 
                inner_abstol::Float64=1e-12,
                inner_reltol::Float64=0.0)
    n, m = size(B)

    function theor_ratio(s, x)
        # x = clamp(x, 0.0, 1.0)
        return  s / sqrt(1 + (s-1) * (x^2))
    end

    function make_block_matrix(m, r)
        @assert m ≥ r "m must be at least r"
        vcat(Matrix(I, r, r), zeros(m-r, r))
    end
    UpId_r = make_block_matrix(m, rank)

    function max_principal_angle_radian(A::AbstractMatrix, B::AbstractMatrix)
        QA = Matrix(qr(A).Q)[:, 1:size(A,2)]
        QB = Matrix(qr(B).Q)[:, 1:size(B,2)]
        svalues = svd(QA' * QB).S
        c = minimum(svalues) # cos(theta_max)
        #c = svalues[end] 
        c = clamp(c, 0.0, 1.0) # avoid LoadError: DomainError with 1.0000000000000002
        return acos(c)
    end

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

    # Initialize loop variables
    # Outer loop
    outer_iters = 0 # The first computation of initial X1 and X2 is not included
    als_error = zeros(Float64, outer_maxiters) # The error between ALS and truncated svd

    bound_ratio = zeros(Float64, outer_maxiters) # The ratio in fonction of sin(theta_max)
    max_angle = zeros(Float64, outer_maxiters) # The angle (in degree) between the two subspaces spanned by SigmaV_truncated' X2 and [I_(rxr), 0_((m-r)xr)]_(mxr)
    upper_error_bound = zeros(Float64, outer_maxiters) # The upper error bound cond(B) * sin(theta_max)

    converged_als = false
    inner_iters = zeros(Int, outer_maxiters) # Total number of inner iterations at each outer step: iters_1 + iters_2
    time_cg = zeros(Float64, outer_maxiters)  # Record time used for each sweep: time_1 + time_2

    # Frobenius norm of the CG residual matrix for each sweep
    cg_last_resnorm_1 = zeros(Float64, outer_maxiters)
    cg_last_resnorm_2 = zeros(Float64, outer_maxiters)

    # Inner loop (Conjugate Gradient)
    Y1 = zeros(n, rank) # Temporary storage for updated X1
    Y2 = zeros(m, rank) # Temporary storage for updated X2

    A1_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X1
    RHS1_cg = zeros(rank, n)  # Right-hand side for CG in updating X1
    A2_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X2
    RHS2_cg = zeros(rank, m)  # Right-hand side for CG in updating X2

    # Initialize parallel processing
    num_threads = Threads.nthreads()
    blocksize1 = ceil(Int, n / num_threads)
    blocksize2 = ceil(Int, m / num_threads)

    # Avoid the first-time overhead in timing
    Threads.@threads for core in 1:num_threads
        nothing
    end

    for s in 1:outer_maxiters

        # Parallelize the inner loop
        # Each thread computes several columns of Y1 (or Y2)

        mul!(A1_cg, X2', X2)     
        mul!(RHS1_cg, X2', B')  
        iters_1 = zeros(Int, n)
        time_1  = zeros(Float64, n) 

        # Frobenius norm of the residual matrix
        cg_last_resvec_1 = zeros(Float64, n) 

        Threads.@threads for core in 1:num_threads
                local_time1_t_j = 0.00
            for j = (core-1)*blocksize1 +1 : min(core*blocksize1, n)
                local_time1_t_j = @elapsed begin 
                    Y1[j, :], ch1_j = cg(A1_cg, @views RHS1_cg[:, j]; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
                end 
                
                # Last residual in cg history - check if resnorm is not empty
                if !isempty(ch1_j.data[:resnorm])
                    cg_last_resvec_1[j] = ch1_j.data[:resnorm][end]
                else
                    cg_last_resvec_1[j] = NaN
                end

                iters_1[j] = ch1_j.iters
                time_1[j]  = local_time1_t_j
            end
        end

        # Orthonormalize X1 
        X1 .= Matrix(qr(Y1).Q)

        mul!(A2_cg, X1', X1)
        mul!(RHS2_cg, X1', B)

        iters_2 = zeros(Int, m)
        time_2  = zeros(Float64, m)

        # Frobenius norm of the residual matrix
        cg_last_resvec_2 = zeros(Float64, m) 

        Threads.@threads for core in 1:num_threads
                    local_time2_t_j = 0.00       
            for j = (core-1)*blocksize2 +1 : min(core*blocksize2, m)
                local_time2_t_j = @elapsed begin 
                    Y2[j, :], ch2_j = cg(A2_cg, @views RHS2_cg[:, j]; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
                end

                # Last residual in cg history - check if resnorm is not empty
                if !isempty(ch2_j.data[:resnorm])
                    cg_last_resvec_2[j] = ch2_j.data[:resnorm][end]
                else
                    cg_last_resvec_2[j] = NaN
                end

                iters_2[j] = ch2_j.iters
                time_2[j] = local_time2_t_j
            end
        end

        # Orthonormalize X2 and compensate X1
        F2 = qr(Y2)
        X2 .= Matrix(F2.Q)
        X1 .= X1 * (F2.R')

        outer_iters += 1

        inner_iters[s] = sum(iters_1) + sum(iters_2)
        als_error[s] = norm(X1 * X2' - B_svd_truncated, 2)
        time_cg[s] = sum(time_1) + sum(time_2)

        cg_last_resnorm_1[s] = sqrt(sum(cg_last_resvec_1.^2))
        cg_last_resnorm_2[s] = sqrt(sum(cg_last_resvec_2.^2))

        #max_angle[s] = acos( opnorm(SigmaV_truncated * X2,2) ) * 180 / pi
        max_angle[s] = max_principal_angle_radian(UpId_r, SigmaV * X2)  # in radian
        bound_ratio[s] = theor_ratio(square_sigma_ratio, sin(max_angle[s]))
        upper_error_bound[s] = cond_B * sin(max_angle[s])

        if als_error[s] <= outer_tol
            println("The matrix ", n, "x", m, " with condition number ", cond(B), " and outer tolerance ", outer_tol, " and inner abstol", inner_abstol, " and inner reltol", inner_reltol," converged at outer iteration: ", s)
            converged_als = true
            break
        end
    end

    # B_als = X1 * X2'
    resize!(als_error, outer_iters)
    resize!(inner_iters, outer_iters)
    resize!(time_cg, outer_iters)
    resize!(cg_last_resnorm_1, outer_iters)
    resize!(cg_last_resnorm_2, outer_iters)
    resize!(bound_ratio, outer_iters)
    resize!(max_angle, outer_iters)
    resize!(upper_error_bound, outer_iters)
    total_time_cg = sum(time_cg) # Total time used in all CG solves
    total_inner_iters = sum(inner_iters)

    X2tX2_fnorm = norm((X2')X2, 2)^2
    println("X2tX2_norm=", X2tX2_fnorm)

    # Compute the angle between the two subspaces spanned by SigmaV_truncated' X2 and [I_(rxr), 0_((m-r)xr)]_(mxr)
    last_theta_max = max_angle[end] * pi / 180
    println("The max principal angle (in degree) between the two subspaces spanned by SigmaV_truncated' X2 and [I_(rxr), 0_((m-r)xr)]_(mxr) is ", last_theta_max)
    last_upper_error_bound = upper_error_bound[end]
    #last_theor_ratio = bound_ratio[end-1]
    last_theor_ratio = length(bound_ratio) ≥ 2 ? bound_ratio[end-1] : NaN
    println("The upper error bound cond(B) * sin(theta_max) is ", last_upper_error_bound)
    println("The ratio in fonction of sin(theta_max) is ", last_theor_ratio)
    

    result = ALSResult(rank, outer_iters, inner_iters, total_inner_iters, als_error, converged_als, time_cg, total_time_cg, outer_tol, inner_abstol, inner_reltol, cg_last_resnorm_1, cg_last_resnorm_2, X2tX2_fnorm, bound_ratio, max_angle, upper_error_bound, last_theta_max, last_upper_error_bound, last_theor_ratio)
    return result
end

#------------------------------------------------------------------------------------------#
# Precompute truncated SVD of B for error computation
function svd_truncated(B::AbstractMatrix, rank::Number)

    B_svd = svd(B, full=false)
    U, S, V = Matrix(B_svd.U), B_svd.S, Matrix(B_svd.V)
    V_truncated = V[:, 1:rank]
    S_truncated = Diagonal(S[1:rank])
    B_svd_truncated = U[:, 1:rank] * S_truncated  * V_truncated'
    svd_error = norm(B - B_svd_truncated, 2)

    singular_values = Vector(B_svd.S)
    ratios = [singular_values[i+1]/singular_values[i] for i in range(1, length(singular_values)-1)]

    #SigmaV_truncated = S_truncated * V_truncated'
    SigmaV = Diagonal(S) * V'


    #square_sigma_ratio = (singular_values[rank+1]/singular_values[rank])^2
    if 1 ≤ rank < length(singular_values)
        square_sigma_ratio = (singular_values[rank+1] / singular_values[rank])^2
    else
        square_sigma_ratio = NaN
    end

    return B_svd_truncated, svd_error, ratios, singular_values, SigmaV, square_sigma_ratio
end


