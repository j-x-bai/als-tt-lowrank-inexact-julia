using IterativeSolvers
using LinearAlgebra
using LinearMaps

# H: n x m -> n x m
# X1: n x r
# X2: m x r
# X = X1 * X2'
# B: n x m
# H(X) = B

# PR : m×r -> m×n,  PR(L) = L*R'
# PR^T : m×n -> m×r, PR^T(X) = X*R
PR!(Y_mn, L_mr, R)  = mul!(Y_mn, L_mr, transpose(R))   # Y = L*Rᵀ
PRt!(Y_mr, X_mn, R) = mul!(Y_mr, X_mn, R)              # Y = X*R

# PL : n×r -> m×n,  PL(R) = L*R'
# PL^T : m×n -> n×r, PL^T(X) = X'*L
PL!(Y_mn, L_mr, R_nr)  = mul!(Y_mn, L_mr, transpose(R_nr))  # = PR!
PLt!(Y_nr, X_mn, L_mr) = mul!(Y_nr, transpose(X_mn), L_mr)

# min_{X1,X2}  1/2 ⟨ H(X1*X2'), X1*X2' ⟩_F - ⟨ B, X1*X2' ⟩_F
# L-step: (PR^T H PR)(X1) = PR^T B
# R-step: (PL^T H PL)(X2) = PL^T B


struct EnergyResult{T <: AbstractFloat}
    X1_sol::Matrix{T}         # m x rank
    X2_sol::Matrix{T}         # n x rank （or rank x n?

    rank::Int
    outer_iters::Int
    inner_iters::Vector{Int}
    total_inner_iters::Int
    

    J_energy::Vector{T}       
    res_fro::Vector{T}   
    J_change::Vector{T}  

    converged_als::Bool
    outer_tol::T
    inner_abstol::T
    inner_reltol::T
end


# H(X) = A * X + X * A' and A=A'
H_op(X, spd_mat) = spd_mat * X + X * spd_mat

function H_op!(HX, X, spd_mat)
    # size(spd_mat) = nxn
    # size(X) = mxn
    # Suppose that m = n
    HX .= spd_mat * X + X * spd_mat  # spd_mat'=spd_mat
    return HX
end

function energy_and_residual(HX, Res, spd_mat, B, X)
    _ = H_op!(HX, X, spd_mat)
    J  = 0.5 * dot(HX, X) - dot(B, X)   # dot(X, Y) = dot(vec(X), vec(Y))
    Res .= HX .- B
    res = norm(Res, 2)     #frobenius                
    return J, res
end

function modified_energy_and_residual(HX, Res, spd_mat, B, X, hat_B_fnrom)
    _ = H_op!(HX, X, spd_mat)
    Res .= HX .- B
    J  = 0.5 * dot(HX, X) - dot(B, X) + 0.5 * (hat_B_fnorm^2)
    res = norm(Res, 2)     #frobenius                
    return J, res
end


function als_2d_energy_spd(A_SPD::AbstractMatrix,
                B::AbstractMatrix, rank::Int;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=1000, 
                outer_tol::Float64=1e-12,
                inner_maxiters::Int=1000, 
                inner_abstol::Float64=1e-12,
                inner_reltol::Float64=0.0)
    m, n = size(B)

    # Initialize starting points
        # In fact, only X2 is needed to initialize, since X1 is updated first.
        if X1 === nothing
            F1 = qr(randn(m, rank))
            X1 = Matrix(F1.Q)#[:, 1:rank]
        else
            F1 = qr(X1)
            X1 = Matrix(F1.Q)#[:, 1:rank]
        end

        if X2 === nothing
            F2 = qr(randn(n, rank))
            X2 = Matrix(F2.Q)#[:, 1:rank]
        else
            F2 = qr(X2)
            X2 = Matrix(F2.Q)#[:, 1:rank]
        end

    outer_iters = 0 # The first computation of initial X1 and X2 is not included
    converged_als = false
    inner_iters = zeros(Int, outer_maxiters)

    # Inner loop (Conjugate Gradient)
    Temp_11 = zeros(rank, rank) # store qr(X2).R'
    Temp_1 = zeros(m, rank) # store X1 * qr(X2).R'

    RHS1 = zeros(m*rank)  # Right-hand side for CG in updating X1
    RHS2 = zeros(n*rank)  # Right-hand side for CG in updating X2

    J = zeros(outer_maxiters)
    J_change = zeros(outer_maxiters)
    res = zeros(outer_maxiters)
    HX = zeros(m,n) # Inplace update
    Res = zeros(m,n) # Inplace update
    X = X1 * X2'
    J[1], res[1] = energy_and_residual(HX, Res, A_SPD, B, X)

    X_mn_1  = zeros(m, n)   #  PR(L)
    HX_mn_1 = zeros(m, n)   #  H(PR(L))
    x_mr_1 = zeros(m*rank)  # Inplace update and initial guess for vec(X1)

    X_mn_2  = zeros(m, n)   #  PR(L)
    HX_mn_2= zeros(m, n)    #  H(PR(L))
    x_nr_2 = zeros(n*rank)  # Inplace update and initial guess for vec(X2)
    for s in 2:outer_maxiters
        outer_iters += 1

        # —— L-step ——  m×r -> m×r
        A1_map_mul! = let X_mn=X_mn_1, HX_mn=HX_mn_1, R=X2 
            function(y::AbstractVector, x::AbstractVector)
                @assert length(x) == m*rank == length(y)

                L = reshape(x, m, rank)
                Y = reshape(y, m, rank)

                PR!(X_mn, L, R)      # X = L*Rᵀ
                H_op!(HX_mn, X_mn, A_SPD)      # HX = H(X)
                PRt!(Y, HX_mn, R)    # Y = HX*R
                return y
            end
        end
        A1 = LinearMap(A1_map_mul!, m*rank; ismutating=true, issymmetric=true) 
        RHS1_mat_view = reshape(RHS1, m, rank)
        mul!(RHS1_mat_view, B, X2)
        _, ch1 = cg!(x_mr_1, A1, RHS1; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
        Y_mr_1 = reshape(x_mr_1, m, rank)
        X1 .= Matrix(qr(Y_mr_1).Q)
        inner_iters[s] = ch1.iters

        # —— R-step ——  n×r <- n×r
        A2_map_mul! = let X_mn=X_mn_2, HX_mn=HX_mn_2, L=X1 
            function(y::AbstractVector, x::AbstractVector)
                @assert length(x) == n*rank == length(y)

                R = reshape(x, n, rank)
                Y = reshape(y, n, rank)

                PL!(X_mn, L, R)      # X = L*Rᵀ
                H_op!(HX_mn, X_mn, A_SPD)      # HX = H(X)
                PLt!(Y, HX_mn, L)    # Y = HXᵀ*L
                return y
            end
        end
        A2 = LinearMap(A2_map_mul!, n*rank; ismutating=true, issymmetric=true)
        RHS2_mat_view = reshape(RHS2, n, rank)
        mul!(RHS2_mat_view, B', X1)
        _, ch2 = cg!(x_nr_2, A2, RHS2; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
        Y_nr_2 = reshape(x_nr_2, n, rank)
        inner_iters[s] += ch2.iters

        # Orthonormalize X2 and compensate X1
        F2 = qr(Y_nr_2)
        X2 .= Matrix(F2.Q)
        Temp_11 .= F2.R'
        Temp_1 .= X1 * Temp_11
        X1 .= Temp_1

        X .= X1 * X2'
        J_s, res_s = energy_and_residual(HX, Res, A_SPD, B, X)
        J[s] = J_s
        res[s] = res_s
        # resrel_cond = res_s ≤ outer_tol * (res_s[s-1] + 1)
        J_change[s] = J_s - J[s-1]
        J_change_cond = abs(J_change[s]) ≤ outer_tol * (abs(J[s-1]) + 1)

        if J_change_cond
            println("The energy solution by min(J) = min{0.5 * dot(HX, X) - dot(B, X)} where m = ",m, " n = ", n, " rank = ", rank, " with outer tolerance ", outer_tol, " and inner abstol", inner_abstol, " and inner reltol", inner_reltol," converged at outer iteration: ", s)
            converged_als = true
            break
        end
        fill!(x_mr_1, 0)
        fill!(x_nr_2, 0)
    end

    resize!(inner_iters, outer_iters)
    resize!(J, outer_iters+1)
    resize!(res, outer_iters+1)
    resize!(J_change, outer_iters)
    total_inner_iters = sum(inner_iters)

    result = EnergyResult(X1, X2, rank, outer_iters, inner_iters, total_inner_iters, J, res, J_change, converged_als, outer_tol, inner_abstol, inner_reltol)
    return result
end

function als_2d_modified_energy_spd(A_SPD::AbstractMatrix,
                B::AbstractMatrix, hat_B_fnrom::Float64,
                rank::Int;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=1000, 
                outer_tol::Float64=1e-12,
                inner_maxiters::Int=1000, 
                inner_abstol::Float64=1e-12,
                inner_reltol::Float64=0.0)
    m, n = size(B)

    # Initialize starting points
        # In fact, only X2 is needed to initialize, since X1 is updated first.
        if X1 === nothing
            F1 = qr(randn(m, rank))
            X1 = Matrix(F1.Q)#[:, 1:rank]
        else
            F1 = qr(X1)
            X1 = Matrix(F1.Q)#[:, 1:rank]
        end

        if X2 === nothing
            F2 = qr(randn(n, rank))
            X2 = Matrix(F2.Q)#[:, 1:rank]
        else
            F2 = qr(X2)
            X2 = Matrix(F2.Q)#[:, 1:rank]
        end

    outer_iters = 0 # The first computation of initial X1 and X2 is not included
    converged_als = false
    inner_iters = zeros(Int, outer_maxiters)

    # Inner loop (Conjugate Gradient)
    Temp_11 = zeros(rank, rank) # store qr(X2).R'
    Temp_1 = zeros(m, rank) # store X1 * qr(X2).R'

    RHS1 = zeros(m*rank)  # Right-hand side for CG in updating X1
    RHS2 = zeros(n*rank)  # Right-hand side for CG in updating X2

    J = zeros(outer_maxiters)
    J_change = zeros(outer_maxiters)
    res = zeros(outer_maxiters)
    HX = zeros(m,n) # Inplace update
    Res = zeros(m,n) # Inplace update
    X = X1 * X2'
    J[1], res[1] = energy_and_residual(HX, Res, A_SPD, B, X)

    X_mn_1  = zeros(m, n)   #  PR(L)
    HX_mn_1 = zeros(m, n)   #  H(PR(L))
    x_mr_1 = zeros(m*rank)  # Inplace update and initial guess for vec(X1)

    X_mn_2  = zeros(m, n)   #  PR(L)
    HX_mn_2= zeros(m, n)    #  H(PR(L))
    x_nr_2 = zeros(n*rank)  # Inplace update and initial guess for vec(X2)
    for s in 2:outer_maxiters
        outer_iters += 1

        # —— L-step ——  m×r -> m×r
        A1_map_mul! = let X_mn=X_mn_1, HX_mn=HX_mn_1, R=X2 
            function(y::AbstractVector, x::AbstractVector)
                @assert length(x) == m*rank == length(y)

                L = reshape(x, m, rank)
                Y = reshape(y, m, rank)

                PR!(X_mn, L, R)      # X = L*Rᵀ
                H_op!(HX_mn, X_mn, A_SPD)      # HX = H(X)
                PRt!(Y, HX_mn, R)    # Y = HX*R
                return y
            end
        end
        A1 = LinearMap(A1_map_mul!, m*rank; ismutating=true, issymmetric=true) 
        RHS1_mat_view = reshape(RHS1, m, rank)
        mul!(RHS1_mat_view, B, X2)
        _, ch1 = cg!(x_mr_1, A1, RHS1; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
        Y_mr_1 = reshape(x_mr_1, m, rank)
        X1 .= Matrix(qr(Y_mr_1).Q)
        inner_iters[s] = ch1.iters

        # —— R-step ——  n×r <- n×r
        A2_map_mul! = let X_mn=X_mn_2, HX_mn=HX_mn_2, L=X1 
            function(y::AbstractVector, x::AbstractVector)
                @assert length(x) == n*rank == length(y)

                R = reshape(x, n, rank)
                Y = reshape(y, n, rank)

                PL!(X_mn, L, R)      # X = L*Rᵀ
                H_op!(HX_mn, X_mn, A_SPD)      # HX = H(X)
                PLt!(Y, HX_mn, L)    # Y = HXᵀ*L
                return y
            end
        end
        A2 = LinearMap(A2_map_mul!, n*rank; ismutating=true, issymmetric=true)
        RHS2_mat_view = reshape(RHS2, n, rank)
        mul!(RHS2_mat_view, B', X1)
        _, ch2 = cg!(x_nr_2, A2, RHS2; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
        Y_nr_2 = reshape(x_nr_2, n, rank)
        inner_iters[s] += ch2.iters

        # Orthonormalize X2 and compensate X1
        F2 = qr(Y_nr_2)
        X2 .= Matrix(F2.Q)
        Temp_11 .= F2.R'
        Temp_1 .= X1 * Temp_11
        X1 .= Temp_1

        X .= X1 * X2'
        J_s, res_s = modified_energy_and_residual(HX, Res, A_SPD, B, X, hat_B_fnrom)
        J[s] = J_s
        res[s] = res_s
        # resrel_cond = res_s ≤ outer_tol * (res_s[s-1] + 1)
        J_change[s] = J_s - J[s-1]
        J_change_cond = abs(J_change[s]) ≤ outer_tol * (abs(J[s-1]) + 1)

        if J_change_cond
            println("The energy solution by min(J) = min{0.5 * dot(HX, X) - dot(B, X)} where m = ",m, " n = ", n, " rank = ", rank, " with outer tolerance ", outer_tol, " and inner abstol", inner_abstol, " and inner reltol", inner_reltol," converged at outer iteration: ", s)
            converged_als = true
            break
        end
        fill!(x_mr_1, 0)
        fill!(x_nr_2, 0)
    end

    resize!(inner_iters, outer_iters)
    resize!(J, outer_iters+1)
    resize!(res, outer_iters+1)
    resize!(J_change, outer_iters)
    total_inner_iters = sum(inner_iters)

    result = EnergyResult(X1, X2, rank, outer_iters, inner_iters, total_inner_iters, J, res, J_change, converged_als, outer_tol, inner_abstol, inner_reltol)
    return result
end

