using Plots
using Printf
using Measures

struct ALSResult 
    X1_sol::Matrix{Float64}
    X2_sol::Matrix{Float64}
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

struct EnergyResult{T <: AbstractFloat}
    X1_sol::Matrix{T}         # m x rank
    X2_sol::Matrix{T}         # n x rank 

    rel_sol_error::Vector{T}  # ‖X^{s} - X_last‖_F / ‖X_last‖_F (like in the paper)

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

function plot_energy_normal_and_Hnorm_normal_solution(
                                            ERes::EnergyResult,          
                                                HRes::ALSResult,             
                                                r::Int,
                                                hat_B_fnorm::Float64,
                                                singular_values::Vector{Float64},
                                                size_B::Tuple{Int, Int},
                                                modified_energy::Bool,
                                                B::AbstractMatrix,            
                                                H_neg_half::AbstractMatrix,   
                                                H_neg::AbstractMatrix;        
                                                cg_method = "IterativeSolve
                                                rs_cg",
                                                name = "A=Id",
                                                outfile = "energy_vs_hnorm"
                                            )

    K = 4  
    K_col = 7
    m,n = size_B
    lay = @layout [grid(K, K_col); T1{0.2h};T2{0.2h}]
    p = plot(layout=lay, size=(8000, K*700+1000), margin=20mm, left_margin=30mm,
             plot_title="Energy vs H-norm | X size = $(size_B) | rank = $r | method = $cg_method | $name")

    idx(row, col) = (row-1)*K_col + col

    #-------------------------------------------------------------------------------------------------------------#
    xE  = 1:(ERes.outer_iters + 1)
    y1E = ERes.J_energy
    y2E = ERes.J_change
    x2E = 1:length(y2E)
    y3E = ERes.res_fro
    y4E = ERes.inner_iters

    if modified_energy
        plot!(p[idx(1,1)], xE, y1E; label="Energy J", legend=false, yscale=:log10,
            xlabel="Outer iteration",
            title="Modified energy in log10: J = 0.5 * dot(HX, X) - dot(B, X) + 0.5 * (hat_B_fnorm^2)at each outer iteration")
    else
        plot!(p[idx(1,1)], xE, y1E; label="Energy J", legend=false,
            xlabel="Outer iteration",
            title="Energy not in log10: J = 0.5 * dot(HX, X) - dot(B, X) at each outer iteration")
    end

    ratio_y1E = [y1E[k]/y1E[k-1] for k in 2:length(y1E)]
    plot!(p[idx(1,2)], xE[2:end], ratio_y1E; legend=false,
          xlabel="Outer iteration", title="Energy: Ratio for J")

    plot!(p[idx(1,3)], x2E, y2E; label="ΔJ", legend=false,
          xlabel="Outer iteration", title="Energy: J change per outer iteration")

    ratio_y2E = [y2E[k]/y2E[k-1] for k in 2:length(y2E)]
    plot!(p[idx(1,4)], x2E[2:end], ratio_y2E; legend=false,
          xlabel="Outer iteration", title="Energy: Ratio for J change")

    if modified_energy
        plot!(p[idx(1,5)], xE, y3E; label="‖HX-B‖_F", legend=false,yscale=:log10,
            xlabel="Outer iteration",
            title="Energy in log10: ‖HX-B‖_F Residual in Frobenius norm")
    else
        plot!(p[idx(1,5)], xE, y3E; label="‖HX-B‖_F", legend=false,
            xlabel="Outer iteration",
            title="Energy not in log10: ‖HX-B‖_F Residual in Frobenius norm")
    end

    ratio_y3E = [y3E[k] / y3E[k-1] for k in 2:length(y3E)]
    plot!(p[idx(1,6)], xE[2:end], ratio_y3E;
        label="ratio(‖Res‖_F)", yaxis=:right, title="Energy: Ratio for ‖HX-B‖_F Residual")

    plot!(p[idx(1,7)], 2:(ERes.outer_iters+1), y4E; legend=false,
          xlabel="Outer iteration", title="Energy: Inner iterations per step")
    #-------------------------------------------------------------------------------------------------------------#
   

    #-------------------------------------------------------------------------------------------------------------#
    xH  = 1:HRes.outer_iters
    y1H = HRes.als_error
    y2H = HRes.inner_iters

    plot!(p[idx(2,1)], xH, y1H; label="ALS error", yscale=:log10, legend=false,
          xlabel="Outer iteration", title="H-norm: ALS error at each outer iteration")

    ratio_y1H = [y1H[k]/y1H[k-1] for k in 2:length(y1H)]
    plot!(p[idx(2,2)], xH[2:end], ratio_y1H; legend=false,
          xlabel="Outer iteration", title="H-norm: Error Ratio vs Theoretical Ratio")

    plot!(p[idx(2,3)], 1:length(y2H), y2H; legend=false,
          xlabel="Outer iteration", title="H-norm: Inner iters")
    #-------------------------------------------------------------------------------------------------------------#


    #-------------------------------------------------------------------------------------------------------------#
    L = min(length(y1E), length(y1H))
    xC = 1:L
    if modified_energy
        plot!(p[idx(3,1)], xC, y1E[1:L]; label="Energy J", legend=:topright,yscale=:log10,
            xlabel="Outer iteration", title="Combined: Energy J & H-norm ALS error in log10")
        plot!(p[idx(3,1)], xC, y1H[1:L]; yscale=:log10,label="H-norm ALS error")
    else
        plot!(p[idx(3,1)], xC, y1E[1:L]; label="Energy J", legend=:topright,
            xlabel="Outer iteration", title="Combined: Energy J & H-norm ALS error not in log10")
        plot!(p[idx(3,1)], xC, y1H[1:L]; label="H-norm ALS error")
    end

    #-------------------------------------------------------------------------------------------------------------#
    # hat_B = reshape(H^{-1/2} * vec(B), m, n)
    vec_B = reshape(B, m*n, 1)
    vec_hat_B = H_neg_half * vec_B
    hat_B = reshape(vec_hat_B, m, n)
    cond_hat_B = cond(hat_B)
    hat_B_svd_truncated, _, _, hat_B_svd_singular_values, _, _ = svd_truncated(hat_B, rank)
    #-------------------------------------------------------------------------------------------------------------#
    hat_X_sol = HRes.X1_sol * (HRes.X2_sol')
    vec_hat_X_sol = reshape(hat_X_sol, m*n, 1)
    vec_X_Hnorm_sol = H_neg_half * vec_hat_X_sol
    X_Hnorm_sol = reshape(vec_X_Hnorm_sol, m, n)
    #-------------------------------------------------------------------------------------------------------------#
    # B_Hnorm_truncated_sol
    vec_B_Hnrom_truncated_sol = H_neg_half * reshape(hat_B_svd_truncated, m*n, 1)
    B_Hnrom_truncated_sol = reshape(vec_B_Hnrom_truncated_sol, m, n)
    B_Hnrom_truncated_sol_svd = svd(B_Hnrom_truncated_sol, full=false)
    B_Hnrom_truncated_sol_svd_singular_values = Vector(B_Hnrom_truncated_sol_svd.S)

    vec_B_Hnrom = H_neg_half * reshape(hat_B, m*n, 1)
    B_Hnrom = reshape(vec_B_Hnrom, m, n)
    B_Hnrom_svd = svd(B_Hnrom,full =false)
    B_Hnrom_singular_values = Vector(B_Hnrom_svd.S)
    #-------------------------------------------------------------------------------------------------------------#
    # Energy sol
    X_energy_sol = ERes.X1_sol * (ERes.X2_sol')

    err_energy_XH        = norm(X_energy_sol - X_Hnorm_sol, 2)
    err_energy_BHtrunc   = norm(X_energy_sol - B_Hnrom_truncated_sol, 2)

    vec_invHB = H_neg * vec_B
    invHB = reshape(vec_invHB, m, n)
    invHB_svd_truncated, _, _, invHB_singular_values, _, _ = svd_truncated(invHB, r)

    err_energy_invHB   = norm(X_energy_sol - invHB_svd_truncated, 2)
    err_XH_invHB       = norm(X_Hnorm_sol - invHB_svd_truncated, 2)
    err_invHB_BHtrunc  = norm(invHB_svd_truncated - B_Hnrom_truncated_sol, 2)

    X_energy_sol_svd = svd(X_energy_sol, full=false)
    X_energy_sol_singular_values = Vector(X_energy_sol_svd.S)
    #-------------------------------------------------------------------------------------------------------------#
    # hat_X_energy_sol
    vec_X_energy_sol = reshape(X_energy_sol, m*n, 1)
    vec_hat_X_energy_sol = H_half * vec_X_energy_sol
    hat_X_energy_sol = reshape(vec_hat_X_energy_sol, m, n) 

    error_hat_X_energy_sol_and_hat_X_sol = norm(hat_X_energy_sol - hat_X_sol, 2)
    error_hat_X_energy_sol_and_hat_B_svd_truncated = norm(hat_X_energy_sol - hat_B_svd_truncated, 2)

    hat_X_energy_sol_svd = svd(hat_X_energy_sol, full=false)
    hat_X_energy_sol_singular_values = Vector(hat_X_energy_sol_svd.S)

    #-------------------------------------------------------------------------------------------------------------#
    r1 = length(hat_X_energy_sol_singular_values)
    x_1 = range(1, r1)
    plot!(p[idx(4,1)], x_1, hat_X_energy_sol_singular_values; label="hat_X_E", legend=:topright, yscale=:log10,
            xlabel="range(1, rank=r)", title="Singular values of hat_X_energy_sol = H^{1/2} * X_E, rank = $(r1)")
    plot!(p[idx(4,7)], x_1, hat_X_energy_sol_singular_values; label="hat_X_E", legend=:topright,yscale=:log10)
    
    r2 = length(hat_B_svd_singular_values)
    x_2 =range(1,r2)
    plot!(p[idx(4,2)], x_2, hat_B_svd_singular_values; label="hat_B", legend=:topright,yscale=:log10,
                xlabel="range(1, rank(hat_B))", title="Singular values of hat_B = H^{-1/2} * B, rank = $(r2)")
    plot!(p[idx(4,7)], x_2, hat_B_svd_singular_values; label="hat_B", legend=:topright,yscale=:log10)
    
    r3 = length(X_energy_sol_singular_values)
    x_3 =range(1,r3)            
    plot!(p[idx(4,3)], x_3, X_energy_sol_singular_values; label="X_E", legend=:topright,yscale=:log10,
            xlabel="range(1, rank=r)", title="Singular values of X_energy_sol = X_E, rank = $(r3)")
    plot!(p[idx(4,7)], x_3, X_energy_sol_singular_values; label="X_E", legend=:topright,yscale=:log10)

    r4 = length(B_Hnrom_singular_values)
    x_4 =range(1,r4) 
    plot!(p[idx(4,4)], x_4, B_Hnrom_singular_values; label="B_Hnrom", legend=:topright,yscale=:log10,
                xlabel="range(1, rank(B_Hnrom))", title="Singular values of B_Hnrom = H^{-1/2} * hat_B =  H^{-1/2} * H^{-1/2} * hat_B, rank = $(r4)")
    plot!(p[idx(4,7)], x_4, B_Hnrom_singular_values; label="B_Hnrom", legend=:topright,yscale=:log10)
                
    r5 = length(B_Hnrom_truncated_sol_svd_singular_values)
    x_5 =range(1,r5)
    plot!(p[idx(4,5)], x_5, B_Hnrom_truncated_sol_svd_singular_values; label="B_Hnrom_truncated", legend=:topright,yscale=:log10,
                xlabel="range(1, rank(B_Hnrom_truncated))", title="Singular values of B_Hnrom_truncated = H^{-1/2} * hat_B_r, rank = $(r5)")
    plot!(p[idx(4,7)], x_5, B_Hnrom_truncated_sol_svd_singular_values; label="B_Hnrom_truncated", legend=:topright,yscale=:log10)

    r6 = length(invHB_singular_values)
    x_6 =range(1,r6)    
    plot!(p[idx(4,6)], x_6, invHB_singular_values; label="invHB", legend=:topright,yscale=:log10,
                xlabel="range(1, rank(invHB))", title="Singular values of invHB = H^{-1} * B, rank = $(r6)")
    plot!(p[idx(4,7)], x_6, invHB_singular_values; label="invHB", legend=:topright,yscale=:log10)



    #-------------------------------------------------------------------------------------------------------------#

    if 1 ≤ r < length(singular_values)
        Sigma_r_square_ratio = (singular_values[r+1] / singular_values[r])^2
        Sigma_r_ratio        =  singular_values[r+1] / singular_values[r]
        singular_values_r    =  singular_values[r]
        singular_values_r1   =  singular_values[r+1]
    else
        Sigma_r_square_ratio = NaN
        Sigma_r_ratio        = NaN
        singular_values_r    = singular_values[r]
        singular_values_r1   = NaN
    end

    # if 1 ≤ r < length(invHB_singular_values)
    #     invHB_Sigma_r_square_ratio = (invHB_singular_values[r+1] / invHB_singular_values[r])^2
    #     invHB_Sigma_r_ratio        =  invHB_singular_values[r+1] / invHB_singular_values[r]
    #     invHB_singular_values_r    =  invHB_singular_values[r]
    #     invHB_singular_values_r1   =  invHB_singular_values[r+1]
    # else
    #     invHB_Sigma_r_square_ratio = NaN
    #     invHB_Sigma_r_ratio        = NaN
    #     invHB_singular_values_r    = invHB_singular_values[r]
    #     invHB_singular_values_r1   = NaN
    # end

    if 1 ≤ r <length(B_Hnrom_singular_values)
        B_Hnrom_Sigma_r_square_ratio = (B_Hnrom_singular_values[r+1] / B_Hnrom_singular_values[r])^2
        B_Hnrom_Sigma_r_ratio        =  B_Hnrom_singular_values[r+1] / B_Hnrom_singular_values[r]
        B_Hnrom_singular_values_r    =  B_Hnrom_singular_values[r]
        B_Hnrom_singular_values_r1   =  B_Hnrom_singular_values[r+1]
    else
        B_Hnrom_Sigma_r_square_ratio = NaN
        B_Hnrom_Sigma_r_ratio        = NaN
        B_Hnrom_singular_values_r    = B_Hnrom_singular_values[r]
        B_Hnrom_singular_values_r1   = NaN
    end





    #-------------------------------------------------------------------------------------------------------------#
    tplot1 = K_col*K + 1
    plot!(p[tplot1], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))

    headers1 = ["type","rank","converged","outer_tol","inner_abstol","inner_reltol",
                "final_metric: J energy | ‖hax_X_sol - hat_B_r‖_F","  outer_iters","total_inner_iters","converged",
                "sigma_r (B_Hnrom = H^{-1/2}*hat_B | hat_B = H^{-1/2}B)","","sigma_{r+1} (B_Hnrom | hat_B)","sigma_{r+1}/sigma_r (B_Hnrom | hat_B)","(sigma_{r+1}/sigma_r)^2 (B_Hnrom_trunc | hat_B)"]

    row_energy = (
        "Energy",
        string(ERes.rank),
        string(ERes.converged_als),
        @sprintf("%.2e", ERes.outer_tol),
        @sprintf("%.2e", ERes.inner_abstol),
        @sprintf("%.2e", ERes.inner_reltol),
        @sprintf("%.3e", last(ERes.J_energy)),
        string(ERes.outer_iters),
        string(ERes.total_inner_iters),
        string(ERes.converged_als),
        @sprintf("%.4e", B_Hnrom_singular_values_r),
        @sprintf("%s", ""),
        @sprintf("%.4e", B_Hnrom_singular_values_r1),
        @sprintf("%.4e", B_Hnrom_Sigma_r_ratio),
        @sprintf("%.4e", B_Hnrom_Sigma_r_square_ratio)
    )

    row_hnorm = (
        "H-norm",
        string(HRes.rank),
        string(HRes.converged_als),
        @sprintf("%.2e", HRes.outer_tol),
        @sprintf("%.2e", HRes.inner_abstol),
        @sprintf("%.2e", HRes.inner_reltol),
        @sprintf("%.3e", last(HRes.als_error)),
        string(HRes.outer_iters),
        string(sum(HRes.inner_iters)),
        string(HRes.converged_als),
        @sprintf("%.4e", singular_values_r),
        @sprintf("%s", ""),
        @sprintf("%.4e", singular_values_r1),
        @sprintf("%.4e", Sigma_r_ratio),
        @sprintf("%.4e", Sigma_r_square_ratio)
    )

    ncols1 = length(headers1)
    xs1 = range(0.01, 0.95; length=ncols1)
    ys1 = range(0.92, 0.60; length=3) 
    for (c, h) in enumerate(headers1)
        annotate!(p[tplot1], xs1[c], ys1[1], text(h, 16, :left))
    end
    for (c, val) in enumerate(row_energy)
        annotate!(p[tplot1], xs1[c], ys1[2], text(val, 16, :left))
    end
    for (c, val) in enumerate(row_hnorm)
        annotate!(p[tplot1], xs1[c], ys1[3], text(val, 16, :left))
    end

    #-------------------------------------------------------------------------------------------------------------#
    headers2 = ["A type",
                "X size",
                "cond(hat_B)",
                "0.5 * (hat_B_fnorm^2)",
                "H norm: last ALS error: ‖hax_X_sol - hat_B_r‖_F, where hat_B_r = trunc_r(hat_B) = trunc_r(H^{-1/2}*B)",
                " ",
                "Energy: last J_energy",
                "Energy: last ‖Res‖_F = ‖ H*X_E - B ‖_F",
                "‖X_E - X_H‖_F, where X_H = H^{-1/2}*hat_X_sol",
                "‖X_E - B_Htrunc‖_F, where B_Htrunc = H^{-1/2}*hat_B_r",
                "‖X_E - (H^{-1}*B)_trunc‖_F, where (H^{-1}*B)_trunc = trunc_r(H^{-1}*B)",
                " ",
                "‖X_H - (H^{-1}*B)_trunc‖_F",
                "‖B_Htrunc - (H^{-1}B)_trunc‖_F",
                "‖H^{1/2}*X_E - hat_X_sol‖_F",
                "‖H^{1/2}*X_E - hat_B_r‖_F"]

    vals2 = (
        string(name),
        string(size_B),
        @sprintf("%.6f", cond_hat_B),
        @sprintf("%.6f", 0.5 * (hat_B_fnorm^2)),
        @sprintf("%.3e", last(HRes.als_error)),
        @sprintf("%s", ""),
        @sprintf("%.3e", last(ERes.J_energy)),
        @sprintf("%.3e", last(ERes.res_fro)),
        @sprintf("%.6g", err_energy_XH),
        @sprintf("%.6g", err_energy_BHtrunc),
        @sprintf("%.6g", err_energy_invHB),
        @sprintf("%s", ""),
        @sprintf("%.6g", err_XH_invHB),
        @sprintf("%.6g", err_invHB_BHtrunc),
        @sprintf("%.6g", error_hat_X_energy_sol_and_hat_X_sol),
        @sprintf("%.6g", error_hat_X_energy_sol_and_hat_B_svd_truncated)
    )

    tplot2 = K_col*K + 2
    plot!(p[tplot2], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols2 = length(headers2)
    xs2 = range(0.01, 0.95; length=ncols2)
    ys2 = range(0.50, 0.28; length=2) 
    for (c, h) in enumerate(headers2)
        annotate!(p[tplot2], xs2[c], ys2[1], text(h, 14, :left))
    end
    for (c, val) in enumerate(vals2)
        annotate!(p[tplot2], xs2[c], ys2[2], text(val, 14, :left))
    end


    # annotate!(p[tplot2], 0.05, 0.16,
    #           text("E = Energy solution, X_H = H-norm solution, B_Htrunc = H-norm back, (H^{-1}B)_trunc = SVD-truncated of H^{-1}B.", 12, :left))

    rowsB, colsB = size_B
    matstr  = "$(rowsB)x$(colsB)"
    Astr = name
    rankstr = "rank_$(r)"
    dir = "data/results_html/$(cg_method)/$(Astr)/$(matstr)/$(rankstr)"
    mkpath(dir)
    if modified_energy
        mestr = "modifiedE"
    else
        mestr="notmodifiedE"
    end
    outfile_html = "$(dir)/$(mestr)_$(outfile)_als_rank_$(r).html"
    savefig(p, outfile_html)
    return outfile_html
end



function plot_energy_normal_and_newton_solution(
                                                NormalRes::EnergyResult,          
                                                NMRes::EnergyResult,             
                                                r::Int,
                                                size_B::Tuple{Int, Int},
                                                modified_energy::Bool;        
                                                cg_method = "IterativeSolvers_cg",
                                                name = "A=Id",
                                                outfile = "energy_normal_vs_energy_newton"
                                            )

    K = 2  
    K_col = 10
    m,n = size_B
    lay = @layout [grid(K, K_col); T1{0.2h}]
    p = plot(layout=lay, size=(10000, K*700+1000), margin=20mm, left_margin=30mm,
             plot_title="Energy vs H-norm | X size = $(size_B) | rank = $r | method = $cg_method | $name")

    idx(row, col) = (row-1)*K_col + col

    #-------------------------------------------------------------------------------------------------------------#
    x_Normal  = 1:(NormalRes.outer_iters + 1)
    y1_Normal = NormalRes.J_energy
    y2_Normal = NormalRes.J_change
    x2_Normal = 1:length(y2_Normal)
    y3_Normal = NormalRes.res_fro
    y4_Normal = NormalRes.inner_iters
    y9_Normal = NormalRes.rel_sol_error
    x9_Normal = 1:length(y9_Normal)

    if modified_energy
        plot!(p[idx(1,1)], x_Normal, y1_Normal; label="Normal", legend=false, yscale=:log10,
            xlabel="Outer iteration",
            title="ALS Normal and ALS Newton: Modified energy in log10: J = 0.5 * dot(HX, X) - dot(B, X) + 0.5 * (hat_B_fnorm^2)at each outer iteration")
    else
        plot!(p[idx(1,1)], x_Normal, y1_Normal; label="Normal", legend=false,
            xlabel="Outer iteration",
            title="ALS Normal and ALS Newton: Energy not in log10: J = 0.5 * dot(HX, X) - dot(B, X) at each outer iteration")
    end

    ratio_y1_Normal = [y1_Normal[k]/y1_Normal[k-1] for k in 2:length(y1_Normal)]
    plot!(p[idx(1,2)], x_Normal[2:end], ratio_y1_Normal; legend=false,label="Normal",
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Energy: Ratio for J")

    plot!(p[idx(1,3)], x2_Normal, y2_Normal; label="Normal ΔJ", legend=false,
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Energy: J change per outer iteration")

    ratio_y2_Normal = [y2_Normal[k]/y2_Normal[k-1] for k in 2:length(y2_Normal)]
    plot!(p[idx(1,4)], x2_Normal[2:end], ratio_y2_Normal; legend=false, label="Normal",
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Energy: Ratio for J change")

    if modified_energy
        plot!(p[idx(1,5)], x_Normal, y3_Normal; label="Normal ‖HX-B‖_F", legend=false,yscale=:log10,
            xlabel="Outer iteration",
            title="ALS Normal and ALS Newton: Energy in log10: ‖HX-B‖_F Residual in Frobenius norm")
    else
        plot!(p[idx(1,5)], x_Normal, y3_Normal; label="Normal ‖HX-B‖_F", legend=false,
            xlabel="Outer iteration",
            title="ALS Normal and ALS Newton: Energy not in log10: ‖HX-B‖_F Residual in Frobenius norm")
    end

    ratio_y3_Normal = [y3_Normal[k] / y3_Normal[k-1] for k in 2:length(y3_Normal)]
    plot!(p[idx(1,6)], x_Normal[2:end], ratio_y3_Normal;
        label="Normal ratio(‖Res‖_F)", yaxis=:right, title="ALS Normal and ALS Newton: Energy: Ratio for ‖HX-B‖_F Residual")

    x7_Normal=2:(NormalRes.outer_iters+1)
    plot!(p[idx(1,7)], x7_Normal, y4_Normal; legend=false,label="Normal",
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Energy: Inner iterations per step")
    
    # Full and half step
    plot!(p[idx(1,8)], x9_Normal, y9_Normal; legend=false,label="Normal", 
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Energy: relative error per full and half step")

    x10_Normal = 1:NormalRes.outer_iters
    y10_Normal = [y9_Normal[2*s] for s in x10_Normal]
    
    # Full step
    plot!(p[idx(1,9)], x10_Normal, y10_Normal; legend=false,label="Normal",yscale=:log10,
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Energy: relative error ‖X^{s} - X_last‖_F / ‖X_last‖_F (like in the paper) per full step")
    ratio_y10_Normal = [y10_Normal[k] / y10_Normal[k-1] for k in x10_Normal[2:end]]
    plot!(p[idx(1,10)], x10_Normal[2:end], ratio_y10_Normal;
        label="Normal ratio rel sol", yaxis=:right, title="ALS Normal and ALS Newton: Energy: Ratio for relative sol error")

    #-------------------------------------------------------------------------------------------------------------#
   

    #-------------------------------------------------------------------------------------------------------------#
    x_NM  = 1:(NMRes.outer_iters + 1)
    y1_NM = NMRes.J_energy
    y2_NM = NMRes.J_change
    x2_NM = 1:length(y2_NM)
    y3_NM = NMRes.res_fro
    y4_NM = NMRes.inner_iters
    y9_NM = NMRes.rel_sol_error
    x9_NM = 1:length(y9_NM)

    if modified_energy
        plot!(p[idx(1,1)], x_NM, y1_NM; label="Newton Energy J", legend=false, yscale=:log10)
    else
        plot!(p[idx(1,1)], x_NM, y1_NM; label="Newton  Energy J", legend=false)
    end

    ratio_y1_NM = [y1_NM[k]/y1_NM[k-1] for k in 2:length(y1_NM)]
    plot!(p[idx(1,2)], x_NM[2:end], ratio_y1_NM; legend=false, label="Newton")

    plot!(p[idx(1,3)], x2_NM, y2_NM; label="Newton ΔJ", legend=false)

    ratio_y2_NM = [y2_NM[k]/y2_NM[k-1] for k in 2:length(y2_NM)]
    plot!(p[idx(1,4)], x2_NM[2:end], ratio_y2_NM; legend=false, label="Newton")

    if modified_energy
        plot!(p[idx(1,5)], x_NM, y3_NM; label="Newton ‖HX-B‖_F", legend=false,yscale=:log10)
    else
        plot!(p[idx(1,5)], x_NM, y3_NM; label="Newton ‖HX-B‖_F", legend=false)
    end

    ratio_y3_NM = [y3_NM[k] / y3_NM[k-1] for k in 2:length(y3_NM)]
    plot!(p[idx(1,6)], x_NM[2:end], ratio_y3_NM;
        label="Newton ratio(‖Res‖_F)")

    x7_NM=2:(NMRes.outer_iters+1)
    plot!(p[idx(1,7)], x7_NM, y4_NM; legend=false, label="Newton")

    # Full and half step
    plot!(p[idx(1,8)], x9_NM, y9_NM; legend=false,label="Newton")

    # Full step
    # x10_NM = 1:NMRes.outer_iters
    # y10_NM = [y9_NM[2*s] for s in x10_NM]
    # plot!(p[idx(1,9)], x10_NM, y10_NM; legend=false,label="NM")
    
    #-------------------------------------------------------------------------------------------------------------#


    #-------------------------------------------------------------------------------------------------------------#
    if modified_energy
        plot!(p[idx(2,1)], x_Normal, y1_Normal; label="ALS Normal: Energy J", legend=:topright,yscale=:log10,
            xlabel="Outer iteration", title="ALS Normal and ALS Newton: Combined: Energy J ALS error in log10")
        plot!(p[idx(2,1)], x_NM, y1_NM; yscale=:log10,label="ALS Newton")
    else
        plot!(p[idx(2,1)], x_Normal, y1_Normal; label="ALS Normal: Energy J", legend=:topright,
            xlabel="Outer iteration", title="ALS Normal and ALS Newton: Combined: Energy J ALS error not in log10")
        plot!(p[idx(2,1)], x_NM, y1_NM; label="ALS Newton ")
    end

    if modified_energy
        plot!(p[idx(2,2)], x_NM, y3_NM; label="ALS Newton: ‖HX-B‖_F", legend=false,yscale=:log10,
            xlabel="Outer iteration",
            title="ALS Normal and ALS Newton: Energy Objective, in log10: ‖HX-B‖_F Residual in Frobenius norm")
        plot!(p[idx(2,2)], x_Normal, y3_Normal; yscale=:log10,label="ALS Normal")
    else
        plot!(p[idx(2,2)], x_NM, y3_NM; label="ALS Newton: ‖HX-B‖_F", legend=false,
            xlabel="Outer iteration",
            title="ALS Normal and ALS Newton: Energy Objective, not in log10: ‖HX-B‖_F Residual in Frobenius norm")
        plot!(p[idx(2,2)], x_Normal, y3_Normal; label="ALS Normal")
    end

    
    #-------------------------------------------------------------------------------------------------------------#
    err_normal_newton = norm(NormalRes.X1_sol*(NormalRes.X2_sol') - NMRes.X1_sol*(NMRes.X2_sol'),2)
    tplot1 = K_col*K + 1
    plot!(p[tplot1], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))

    headers1 = ["A type","ALS type"," X size", "rank","converged","outer_tol","inner_abstol","inner_reltol",
                "last J energy", "last ‖hax_X_sol - hat_B_r‖_F","  outer_iters", "total_inner_iters", "The error of last solution between ALS Normal and ALS Newton"]

    row_energy = (
        name,
        "ALS Normal",
        string(size(NormalRes.X1_sol,1)," x ",size(NormalRes.X2_sol,1)),
        string(NormalRes.rank),
        string(NormalRes.converged_als),
        @sprintf("%.2e", NormalRes.outer_tol),
        @sprintf("%.2e", NormalRes.inner_abstol),
        @sprintf("%.2e", NormalRes.inner_reltol),
        @sprintf("%.3e", last(NormalRes.J_energy)),
        @sprintf("%.3e", last(NormalRes.res_fro)),
        string(NormalRes.outer_iters),
        string(NormalRes.total_inner_iters),
        @sprintf("%.4e", err_normal_newton)
    )

    row_hnorm = (
        name,
        "ALS Newton",
        string(size(NMRes.X1_sol,1)," x ",size(NMRes.X2_sol,1)),
        string(NMRes.rank),
        string(NMRes.converged_als),
        @sprintf("%.2e", NMRes.outer_tol),
        @sprintf("%.2e", NMRes.inner_abstol),
        @sprintf("%.2e", NMRes.inner_reltol),
        @sprintf("%.3e", last(NMRes.J_energy)),
        @sprintf("%.3e", last(NMRes.res_fro)),
        string(NMRes.outer_iters),
        string(NMRes.total_inner_iters),
        @sprintf("%.4e", err_normal_newton)
    )

    ncols1 = length(headers1)
    xs1 = range(0.01, 0.95; length=ncols1)
    ys1 = range(0.92, 0.60; length=3) 
    for (c, h) in enumerate(headers1)
        annotate!(p[tplot1], xs1[c], ys1[1], text(h, 16, :left))
    end
    for (c, val) in enumerate(row_energy)
        annotate!(p[tplot1], xs1[c], ys1[2], text(val, 16, :left))
    end
    for (c, val) in enumerate(row_hnorm)
        annotate!(p[tplot1], xs1[c], ys1[3], text(val, 16, :left))
    end

    #-------------------------------------------------------------------------------------------------------------#

    rowsB, colsB = size_B
    matstr  = "$(rowsB)x$(colsB)"
    Astr = name
    rankstr = "rank_$(r)"
    dir = "data/results_html/$(cg_method)/Energy_ALS_Normal_and_ALS_Newton/$(Astr)/$(matstr)/$(rankstr)"
    mkpath(dir)
    if modified_energy
        mestr = "modifiedE"
    else
        mestr="notmodifiedE"
    end
    outfile_html = "$(dir)/$(mestr)_$(outfile)_als_rank_$(r).html"
    savefig(p, outfile_html)
    return outfile_html
end