using IterativeSolvers
using LinearAlgebra
# using BenchmarkTools
using Base.Threads
using JLD2
using Plots

# struct ALSResult 
#     rank::Int
#     outer_iters::Int
#     inner_iters::Vector{Int}
#     total_inner_iters::Int
#     als_error::Vector{Float64}
#     converged_als::Bool
#     time_cg::Vector{Float64}
#     total_time_cg::Float64
#     outer_tol::Float64
#     inner_abstol::Float64
#     inner_reltol::Float64
#     cg_last_resnorm_1::Vector{Float64}
#     cg_last_resnorm_2::Vector{Float64}
#     X2tX2_fnorm::Float64
#     bound_ratio::Vector{Float64}
#     max_angle::Vector{Float64} # radians
#     upper_error_bound::Vector{Float64}
#     last_theta_max::Float64
#     last_upper_error_bound::Float64
#     last_theor_ratio::Float64
# end

# H: n x m -> n x m
# X1: n x r
# X2: m x r
# X = X1 * X2'
# B: n x m
# H(X) = B

# PR : m×r -> m×n,  PR(L) = L*R'
# PR^T : m×n -> m×r, PR^T(X) = X*R
PR(Y_mn, L_mr, R)  = mul!(Y_mn, L_mr, transpose(R))   # Y = L*Rᵀ
PRt(Y_mr, X_mn, R) = mul!(Y_mr, X_mn, R)              # Y = X*R

# PL : n×r -> m×n,  PL(R) = L*R'
PL(R, L) = L * R'
# PL^T : m×n -> n×r, PL^T(X) = X'*L
PLt(X, L) = X' * L


# min_{X1,X2}  1/2 ⟨ H(X1*X2'), X1*X2' ⟩_F - ⟨ B, X1*X2' ⟩_F
# L-step: (PR^T H PR)(X1) = PR^T B
# R-step: (PL^T H PL)(X2) = PL^T B

struct HnormResult 
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

function energy_and_residual(HX, Res, H, B, X; eps=1e-16)
    HX .= H(X)
    Res  .= HX .- B
    J  = 0.5 * dot(HX, X) - dot(B, X)   # dot(X, Y) = dot(vec(X), vec(Y))
    res = norm(Res, 2)     #frobenius                  
    return J, res
end


function als_2d_energy(H::AbstractMatrix,
                B::AbstractMatrix, rank::Int;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=10000, 
                outer_tol::Float64=1e-12,
                inner_maxiters::Int=10000, 
                inner_abstol::Float64=1e-12,
                inner_reltol::Float64=0.0)
    n, m = size(B)
    B_fnorm = norm(B, 2)

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

    outer_iters = 0 # The first computation of initial X1 and X2 is not included
    converged_als = false

    # Inner loop (Conjugate Gradient)
    Y1 = zeros(n, rank) # Temporary storage for updated X1
    Y2 = zeros(m, rank) # Temporary storage for updated X2

    #A1_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X1 ?????????
    RHS1_cg = zeros(n, rank)  # Right-hand side for CG in updating X1
    #A2_cg = zeros(rank, rank) # Coefficient matrix for CG in updating X2
    RHS2_cg = zeros(m,rank)  # Right-hand side for CG in updating X2

    J = zeros(outer_iters)
    res = zeros(outer_iters)
    HX = zeros(m,n)
    Res = zeros(m,n)

    X_mn  = zeros(m, n)   # 工作缓冲，存 PR(L)
    HX_mn = zeros(m, n)   # 工作缓冲，存 H(PR(L))
    Y_mr  = zeros(m, r)   # 工作缓冲，存 PR^T(H(...))

    for s in 1:outer_maxiters

        # —— L-step ——  m×r -> m×r
        A1_cg = Lx -> PRt(H(PR(Lx, X2)), X2)  #closure (P_R^T H P_R)(Lx)
        RHS1_cg .= PRt(B, X2)                  # P_R^T B
        Y1, ch1= cg(A1_cg, RHS1_cg; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)
        X1 .= Matrix(qr(Y1).Q)

        # —— R-step ——  n×r <- n×r
        AopR = Rx -> PLt(H(PL(Rx, X1)), X1)  # (P_L^T H P_L)(Rx)
        RHS2_cg .= PLt(B, X1)                 # P_L^T B
        Y2, ch2 = cg(A2_cg, RHS2_cg; abstol=inner_abstol, maxiter=inner_maxiters, log=true, reltol=inner_reltol)

        # Orthonormalize X2 and compensate X1
        F2 = qr(Y2)
        X2 .= Matrix(F2.Q)
        X1 .= X1 * (F2.R')

        outer_iters += 1
        J_s, res_s = energy_and_residual(HX, Res, H, B, X; eps=1e-16)
        J[s] = J_s
        res[s] = res_s
        #resrel_cond = res_s ≤ outer_tol * (res_s[s-1] + 1)
        J_rel_cond = J_s ≤ outer_tol * (J_s[s-1] + 1)

        if J_rel_cond
            println("The HX=B ", " outer tolerance ", outer_tol, " and inner abstol", inner_abstol, " and inner reltol", inner_reltol," converged at outer iteration: ", s)
            converged_als = true
            break
        end
    end




    result = ALSResult(rank, outer_iters, inner_iters, total_inner_iters, als_error, converged_als, time_cg, total_time_cg, outer_tol, inner_abstol, inner_reltol, cg_last_resnorm_1, cg_last_resnorm_2, X2tX2_fnorm, bound_ratio, max_angle, upper_error_bound, last_theta_max, last_upper_error_bound, last_theor_ratio)
    return result
end



using LinearMaps, IterativeSolvers

# 预分配工作区
X_mn  = zeros(m, n)
HX_mn = similar(X_mn)
Y_mr  = zeros(m, r)

# 就地乘法： y = A1*x  （x,y 均为长度 m*r 的向量）
function A1_mul!(y::AbstractVector, x::AbstractVector)
    @assert length(x) == m*r == length(y)
    Lx = reshape(x, m, r)
    Y  = reshape(y, m, r)
    PR!(X_mn, Lx, R)        # X_mn = Lx * Rᵀ
    H!(HX_mn, X_mn)         # HX_mn = H(X_mn)
    PRt!(Y, HX_mn, R)       # Y = HX_mn * R
    return y
end

A1 = LinearMap(A1_mul!, m*r; ismutating=true, issymmetric=true)
RHS = vec(B*R)  # 右端仍然是向量
x, ch = cg(A1, RHS; reltol=1e-8, maxiter=2000, log=true)
Lhat = reshape(x, m, r)
