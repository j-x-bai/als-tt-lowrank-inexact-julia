using Plots
using Printf
using Measures 

function plot_energy_and_Hnorm_solution(
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

    K = 3  
    K_col = 7
    m,n = size_B
    lay = @layout [grid(K, K_col); T1{0.2h};T2{0.2h}]
    p = plot(layout=lay, size=(8000, K*700+1000), margin=20mm, left_margin=30mm,
             plot_title="Energy vs H-norm | X size = $(size_B) | rank = $r | method = $cg_method | $name")

    idx(row, col) = (row-1)*K_col + col

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

    # hat_B = reshape(H^{-1/2} * vec(B), m, n)
    vec_B = reshape(B, m*n, 1)
    vec_hat_B = H_neg_half * vec_B
    hat_B = reshape(vec_hat_B, m, n)
    cond_hat_B = cond(hat_B)
    hat_B_svd_truncated, _, _, _, _, _ = svd_truncated(hat_B, rank)

    hat_X_sol = HRes.X1_sol * (HRes.X2_sol')
    vec_hat_X_sol = reshape(hat_X_sol, m*n, 1)
    vec_X_Hnorm_sol = H_neg_half * vec_hat_X_sol
    X_Hnorm_sol = reshape(vec_X_Hnorm_sol, m, n)

    # B_Hnorm_truncated_sol
    vec_B_Hnrom_truncated_sol = H_neg_half * reshape(hat_B_svd_truncated, m*n, 1)
    B_Hnrom_truncated_sol = reshape(vec_B_Hnrom_truncated_sol, m, n)

    # B_Hnorm
    vec_B_Hnrom = H_neg_half * reshape(hat_B, m*n, 1)
    B_Hnrom = reshape(vec_B_Hnrom, m, n)
    B_Hnrom_svd = svd(B_Hnrom,full =false)
    B_Hnrom_singular_values = Vector(B_Hnrom_svd.S)

    # Energy sol
    X_energy_sol = ERes.X1_sol * (ERes.X2_sol')

    err_energy_XH        = norm(X_energy_sol - X_Hnorm_sol, 2)
    err_energy_BHtrunc   = norm(X_energy_sol - B_Hnrom_truncated_sol, 2)

    vec_invHB = H_neg * vec_B
    invHB = reshape(vec_invHB, m, n)
    invHB_svd_truncated, _, _, _, _, _ = svd_truncated(invHB, r)

    err_energy_invHB   = norm(X_energy_sol - invHB_svd_truncated, 2)
    err_XH_invHB       = norm(X_Hnorm_sol - invHB_svd_truncated, 2)
    err_invHB_BHtrunc  = norm(invHB_svd_truncated - B_Hnrom_truncated_sol, 2)


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
                "‖B_Htrunc - (H^{-1}B)_trunc ‖_F"]

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
        @sprintf("%.6g", err_invHB_BHtrunc)
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

    rowsB, colsB = size_B
    matstr  = "$(rowsB)x$(colsB)"
    Astr = name
    rankstr = "rank_$(r)"
    dir = "data_EH/results_html/$(cg_method)/$(Astr)/$(matstr)/$(rankstr)"
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