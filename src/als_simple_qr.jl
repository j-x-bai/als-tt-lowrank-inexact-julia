using IterativeSolvers
using LinearAlgebra

function als_2d_qr(B::AbstractMatrix, r::Number;
                X1 = nothing, X2 = nothing,
                outer_maxiters::Int=100, outer_tol::Float64=1e-8,
                # inner_method::Symbol=:cg, 
                inner_maxiters::Int=100, inner_tol::Float64=1e-8)
    n, m = size(B)
    
    if X1 === nothing
        QR1 = qr(randn(n, r))
        X1 = Matrix(QR1.Q)
    else
        QR1 = qr(X1)
        X1 = Matrix(QR1.Q)
    end

    if X2 === nothing
        QR2 = qr(randn(m, r))
        X2 = Matrix(QR2.Q)
    else
        QR2 = qr(X2)
        X2 = Matrix(QR2.Q)
    end

    error = zeros(outer_maxiters)
    error[1] = norm(X1 * X2' - B, 2)
    Y1 = zeros(n, r)
    Y2 = zeros(m, r)
    converged_als = false

    # id = Matrix{Float64}(I, rank, rank)
  
    for s in 1:outer_maxiters
        
        blocksize1 = ceil(Int, n / num_threads)
        Threads.@threads for core in 1:num_threads
            for j = (core-1)*blocksize1 +1 : min(core*blocksize1, n)
                Y1[j, :] = cg(X2' * X2, (X2' * B')[:, j]; abstol=inner_tol, maxiter=inner_maxiters)
            end
        end

        QR1 = qr(Y1)
        X1 = Matrix(QR1.Q)

        blocksize2 = ceil(Int, m / num_threads)
        Threads.@threads for core in 1:num_threads
            for j = (core-1)*blocksize2 +1 : min(core*blocksize2, m)
                Y2[j, :] = cg(X1' * X1, (X1' * B)[:, j]; abstol=inner_tol, maxiter=inner_maxiters)
            end
        end
        QR2 = qr(Y2)

        X2 = Matrix(QR2.Q)
        X1 = X1 * (QR2.R')

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

    return U[:, 1:rank] * Diagonal(S[1:rank]) * V[:, 1:rank]', S
end
