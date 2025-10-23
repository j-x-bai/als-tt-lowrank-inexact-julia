using JLD2
using Plots
using Printf
using Measures
using Base.Threads
plotlyjs()
#gr()


function save_rank_panel_pdf(r::Int, resvec::Vector{ALSResult}, 
                             svd_ratios::Vector{Float64},singular_values::Vector{Float64},
                             size_B::Tuple{Int, Int}, cond_B::Float64;
                             cg_method = "IterativeSolvers_cg",
                             als_method = "ALS_Normal",
                             names = ["outer=$(resvec[i].outer_tol), inner=$(resvec[i].inner_abstol)" for i in range(1, length(resvec))],
                             outfile = "exact")

    K = length(resvec)

    lay = @layout [grid(K, 6); T{0.05h}]
    p = plot(layout=lay, size=(7000, K*700), margin=20mm, left_margin=30mm, 
             plot_title="Exact ALS Rank Panel | matrix size = $(size_B) cond = $cond_B |  rank = $r |  method = $cg_method ")

    idx(row, col) = (row-1)*6 + col
    last_ratio = zeros(K)

    for j in 1:K
        res = resvec[j]
        x  = range(1, res.outer_iters)
        y1 = res.als_error
        y2 = res.inner_iters
        y3 = res.time_cg
        y4 = res.cg_last_resnorm_1
        y5 = res.cg_last_resnorm_2
        
        y6 = res.bound_ratio
        y7 = res.max_angle  * pi / 180
        y8 = res.upper_error_bound

        plot!(p[idx(j,1)], x, y1; label="Frobenius, original results", yscale=:log10, legend=false,
              xlabel="Outer iteration", title="Original error in Frobenius norm vs cond_B * sin(max_theta) * B_fnorm in 2-norm at each outer iteration($(names[j]))")
        plot!(p[idx(j,1)], x, y8;label="Upper error bound in opnorm-2")

        # Compute error ratio 
        ratio = [y1[k]/y1[k-1] for k in range(2,length(y1))]
        if !isempty(ratio)
            last_ratio[j] = ratio[end]
        else
            last_ratio[j] = NaN
        end
        # print("length(ratio) = ", length(ratio), "\n")
        plot!(p[idx(j,2)], x[2:end], ratio; xlabel="Outer iteration",title="Error Ratio vs Theoretical Ratio($(names[j]))", legend=false)
        plot!(p[idx(j,2)], x, y6;label="ratio of upper theoretical bound")

        plot!(p[idx(j,3)], x, y2; legend=false,
        xlabel="Outer iteration", title="Inner iters ($(names[j]))")
        plot!(p[idx(j,4)], x, y3; legend=false,
                xlabel="Outer iteration", title="Time ($(names[j]))")

        plot!(p[idx(j,5)], x, y4;label="left", 
              xlabel="Outer iteration", title="CG residual matrix in frobenius norm ($(names[j]))")
        plot!(p[idx(j,5)], x, y5;label="right")

        plot!(p[idx(j,6)], x, y7; legend=false,
              xlabel="Outer iteration", title="Max principal angle between subspaces in degree ($(names[j]))")
        
    end

    if 1 ≤ r < length(singular_values)
            Sigma_r_square_ratio = (singular_values[r+1] / singular_values[r])^2
            Sigma_r_ratio = singular_values[r+1] / singular_values[r]
            singular_values_r = singular_values[r]
            singular_values_r1 = singular_values[r+1]
    else
            Sigma_r_square_ratio = NaN
            Sigma_r_ratio = NaN
            singular_values_r = singular_values[r]
            singular_values_r1 = NaN
    end

    headers = ["rank","outer_tol", "inner_abstol", "inner_reltol","final_error", "total_time", "total_inner_iters", "outer_iters", "converged", "final_ratio", "X2tX2_fnorm", "last_theoretical_error_bound", "last_theta_max", "last_theoretical_ratio", "sigma_r", "sigma_{r+1}", "sigma_{r+1}/sigma_r", "(sigma_{r+1}/sigma_r)^2"]
    rows = [(
        string(resvec[j].rank),
        @sprintf("%.2e", resvec[j].outer_tol),
        @sprintf("%.2e", resvec[j].inner_abstol),
        @sprintf("%.2e", resvec[j].inner_reltol),
        @sprintf("%.3e", last(resvec[j].als_error)),
        @sprintf("%.3e", resvec[j].total_time_cg),
        string(resvec[j].total_inner_iters),
        string(resvec[j].outer_iters),
        string(resvec[j].converged_als),
        @sprintf("%.4f", last_ratio[j]),
        @sprintf("%.3e", resvec[j].X2tX2_fnorm),
        @sprintf("%.3e", resvec[j].last_upper_error_bound),
        @sprintf("%.3f", resvec[j].last_theta_max),
        @sprintf("%.4e", resvec[j].last_theor_ratio),
        @sprintf("%.4e", singular_values_r),
        @sprintf("%.4e", singular_values_r1),
        @sprintf("%.4e", Sigma_r_ratio),
        @sprintf("%.4e", Sigma_r_square_ratio)
    ) for j in 1:K]

    tplot = 6K + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        nrows = length(rows) + 1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in range(1, length(rows))
            vals = rows[rrow]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1], text(vals[c], 16, :black, :center))
            end
        end
        annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end
    rows, cols = size_B
    matstr = "$(rows)x$(cols)"
    condstr = @sprintf("cond%.2f", cond_B)
    rankstr = "rank_$(r)"

    dir = "data/results_html/$(cg_method)/$(als_method)/$(matstr)_$(condstr)/$(rankstr)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(outfile)_als_rank_$(r).html"
    savefig(p, outfile_html)
    # savehtml(p, "als_fast_test.html"; include_plotlyjs="cdn")

    return outfile_html
end


function save_relaxation_panel_pdf(
                                    r::Int,
                                    resvec_inner::Vector{ALSResult},
                                    svd_ratios::Vector{Float64},
                                    singular_values::Vector{Float64},
                                    outer_tol::Float64,
                                    size_B::Tuple{Int, Int}, cond_B::Float64;
                                    cg_method = "IterativeSolver_cg",
                                    als_method = "ALS Normal",
                                    outfile::AbstractString = "als_relaxation",
                                )

    lay = @layout [grid(7,1)]
    p = plot(layout=lay, size=(5000, 7000), margin=10mm, left_margin=20mm,
             plot_title="ALS Relaxation Panel |matrix size = $(size_B) cond = $cond_B |  rank = $r| outer_tol=$(@sprintf("%.1e", outer_tol)) | method = $cg_method ")

    vec_inner_tol = [res.inner_abstol for res in resvec_inner]
    vec_outer_iters = [res.outer_iters for res in resvec_inner]

    allpos_errors = !isempty(resvec_inner) && all(res -> !isempty(res.als_error) && all(>(0), res.als_error), resvec_inner)

    plot!(p[1], title="Error at each outer step (outer_tol=$(@sprintf("%.1e", outer_tol)))",
          xlabel="Outer step", ylabel="Error", legend=:topright,
          yscale = allpos_errors ? :log10 : :identity)

    plot!(p[2], title="Time at each outer step (outer_tol=$(@sprintf("%.1e", outer_tol)))",
          xlabel="Outer step", ylabel="Time (s)", legend=:topright)

    plot!(p[3], title="Number of inner iterations at each outer step (fixed outer_tol=$(@sprintf("%.1e", outer_tol)))",
        xlabel="Outer step", ylabel="Inner iters", legend=:topright)

    plot!(p[4], title="Number of outer iterations vs inner tol (fixed outer_tol=$(@sprintf("%.1e", outer_tol)))",
          xlabel="Inner tol", ylabel="Outer iters", legend=:topright, xscale=:log10)

    plot!(p[5], title="Number of total inner iterations vs inner tol (fixed outer_tol=$(@sprintf("%.1e", outer_tol)))",
          xlabel="Inner tol", ylabel="Inner iters", legend=:topright, xscale=:log10)
    
    plot!(p[6], title="Number of total time vs inner tol (fixed outer_tol=$(@sprintf("%.1e", outer_tol)))",
          xlabel="Inner tol", ylabel="Total time (s)", legend=:topright, xscale=:log10)

    for res in resvec_inner
        lab = @sprintf("inner=%.1e", res.inner_abstol)

        # Error series  
        y1 = res.als_error
        if !isempty(y1)
            x1 = 1:length(y1)
            plot!(p, x1, y1; label=lab, subplot=1)
        end

        # Time series
        y2 = res.time_cg
        if !isempty(y2)
            x2 = 1:length(y2)
            plot!(p, x2, y2; label=lab, subplot=2)
        end

        # Inner iters series
        y3 = res.inner_iters
        if !isempty(y3)
            x3 = 1:length(y3)
            plot!(p, x3, y3; label=lab, subplot=3)
        end
    end

    if !isempty(vec_inner_tol) && !isempty(vec_outer_iters)
        plot!(p, vec_inner_tol, vec_outer_iters;
              label="outer tol = $(@sprintf("%.1e", outer_tol))",
              subplot=4)
        plot!(p, vec_inner_tol, [res.total_inner_iters for res in resvec_inner];
              label="outer tol = $(@sprintf("%.1e", outer_tol))",
              subplot=5)
        plot!(p, vec_inner_tol, [res.total_time_cg for res in resvec_inner];
              label="outer tol = $(@sprintf("%.1e", outer_tol))",
              subplot=6)

        # log ticks on x for subplots 4-6
        if all(>(0), vec_inner_tol)
            lo = minimum(vec_inner_tol); hi = maximum(vec_inner_tol)
            emin = floor(Int, log10(lo)); emax = ceil(Int, log10(hi))
            if emin == emax
                emin -= 1; emax += 1
            end
            exps = emin:emax
            tickvals = 10.0 .^ exps
            ticklabels = [@sprintf("1e%d", e) for e in exps]
            plot!(p[4]; xticks=(tickvals, ticklabels), xlims=(minimum(tickvals), maximum(tickvals)))
            plot!(p[5]; xticks=(tickvals, ticklabels), xlims=(minimum(tickvals), maximum(tickvals)))
            plot!(p[6]; xticks=(tickvals, ticklabels), xlims=(minimum(tickvals), maximum(tickvals)))
        end
    end

    # Subplot 7: configurations table
    tplt_cfg = 7
    plot!(p[tplt_cfg], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))


    if 1 ≤ r < length(singular_values)
            Sigma_r_square_ratio = (singular_values[r+1] / singular_values[r])^2
            Sigma_r_ratio = singular_values[r+1] / singular_values[r]
            singular_values_r = singular_values[r]
            singular_values_r1 = singular_values[r+1]
    else
            Sigma_r_square_ratio = NaN
            Sigma_r_ratio = NaN
            singular_values_r = singular_values[r]
            singular_values_r1 = NaN
    end

    headers = ["rank","inner_abstol", "inner_reltol", "outer_tol", "final_err", "total_time", "total_inner_iters","outer_iters", "converged", "X2tX2_fnorm", "last_theoretical_error_bound", "last_theta_max", "last_theoretical_ratio", "sigma_r", "sigma_{r+1}", "sigma_{r+1}/sigma_r", "(sigma_{r+1}/sigma_r)^2"]
    rows_tbl = [(
        string(res.rank),
        @sprintf("%.2e", res.inner_abstol),
        @sprintf("%.2e", res.inner_reltol),
        @sprintf("%.2e", res.outer_tol),
        @sprintf("%.3e", last(res.als_error)),
        @sprintf("%.3e", res.total_time_cg),
        string(res.total_inner_iters),
        string(res.outer_iters),
        string(res.converged_als),
        string("%.3e", res.X2tX2_fnorm),
        @sprintf("%.3e", res.last_upper_error_bound),
        @sprintf("%.3f", res.last_theta_max),
        @sprintf("%.4e", res.last_theor_ratio),
        @sprintf("%.4e", singular_values_r),
        @sprintf("%.4e", singular_values_r1),
        @sprintf("%.4e", Sigma_r_ratio),
        @sprintf("%.4e", Sigma_r_square_ratio)
    ) for res in resvec_inner]

    if isempty(rows_tbl)
        annotate!(p[tplt_cfg], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        ncols = length(headers)
        nrows = length(rows_tbl) + 1
        xs = range(0.05, 0.95; length=ncols)
        ys = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplt_cfg], xs[c], ys[1], text(h, 16, :left))
        end
        for (ridx, row) in enumerate(rows_tbl)
            for (c, val) in enumerate(row)
                annotate!(p[tplt_cfg], xs[c], ys[ridx+1], text(val, 16, :left))
            end
        end
    end

    # Output path
    rowsB, colsB = size_B
    matstr = "$(rowsB)x$(colsB)"
    condstr = @sprintf("cond%.2f", cond_B)
    rankstr = "rank_$(r)"
    outertolstr = @sprintf("outertol_%1.0e", outer_tol)
    dir = "data/results_html/$(cg_method)/$(als_method)/$(matstr)_$(condstr)/$(rankstr)/$(outertolstr)"
    mkpath(dir)
    # outpath = "$(dir)/$(outfile)_als_rank_$(r).pdf"
    # outpath_pdf = "$(dir)/$(outfile)_als_rank_$(r).pdf"
    # savefig(p, outpath_pdf)
    outpath_html = "$(dir)/$(outfile)_als_rank_$(r).html"
    savefig(p, outpath_html)
    return outpath_html
end


function save_singular_values_csv(r::Int, singular_values::Vector{Float64}, svd_ratios::Vector{Float64},
                                size_B::Tuple{Int, Int}, cond_B::Float64
                                ; cg_method = "IterativeSolvers_cg",
                                outfile="singular_values_table")
    rows, cols = size_B
    matstr  = "$(rows)x$(cols)"
    condstr = @sprintf("cond%.2f", cond_B)
    rankstr = "rank_$(r)"
    dir = "data/results_html/$(cg_method)/singular_values/$(matstr)_$(condstr)/$(rankstr)"
    mkpath(dir)

    path = "$(dir)/$(outfile)_als_rank_$(r).csv"
    open(path, "w") do io
        println(io, "i,sigma_i,sigma_{i+1}/sigma_i,(sigma_{i+1}/sigma_i)^2,sigma_i * (sigma_{i+1}/sigma_i)")
        n = length(singular_values)
        for i in 1:n
            sigma = @sprintf("%.3e", singular_values[i])
            ratio = (i < n) ? @sprintf("%.3e", svd_ratios[i]) : "-"
            ratio_square = (i < n) ? @sprintf("%.3e", svd_ratios[i]^2) : "-"
            sigma_x_ratio = (i < n) ? @sprintf("%.3e", singular_values[i]*svd_ratios[i]) : "-"
            println(io, "$(i),$(sigma),$(ratio),$(ratio_square),$(sigma_x_ratio)")
        end
    end
    return path
end



function compare_2d_Id_als_normal_and_NM_exact(r::Int, resvec_Normal::Vector{ALSResult}, 
                             resvec_NM::Vector{ALSResult}, 
                             svd_ratios::Vector{Float64},singular_values::Vector{Float64},
                             size_B::Tuple{Int, Int}, cond_B::Float64;
                             cg_method = "IterativeSolvers_cg",
                             names = ["outer=$(resvec_Normal[i].outer_tol), inner=$(resvec_Normal[i].inner_abstol)" for i in range(1, length(resvec_Normal))],
                             outfile = "exact_ALS_Normal_and_ALS_Newton")

    K = length(resvec_Normal)
    Kcol = 9

    lay = @layout [grid(K, Kcol); T{0.1h}]
    p = plot(layout=lay, size=(10000, K*700+1000), margin=20mm, left_margin=30mm, 
             plot_title="Exact ALS Rank Panel: ALS Normal and ALS Newton | matrix size = $(size_B) cond = $cond_B |  rank = $r |  method = $cg_method ")

    idx(row, col) = (row-1)*Kcol + col
    last_ratio_Normal = zeros(K)
    last_ratio_NM = zeros(K)
    error_norm_sol_Normal_Newton = zeros(K)
    error_norm_X1_sol_Normal_Newton = zeros(K)
    error_norm_X2_sol_Normal_Newton = zeros(K)
    error_mat_sol_Normal_Newton = zeros(size_B)
    error_mat_X1_sol_Normal_Newton = zeros(size_B[1], r)
    error_mat_X2_sol_Normal_Newton = zeros(size_B[2], r)

    for j in 1:K
        res_Normal = resvec_Normal[j]
        x_Normal  = range(1, res_Normal.outer_iters)
        y1_Normal = res_Normal.als_error
        y2_Normal = res_Normal.inner_iters
        y3_Normal = res_Normal.time_cg
        y4_Normal = res_Normal.cg_last_resnorm_1
        y5_Normal = res_Normal.cg_last_resnorm_2
        
        # y6_Normal = res_Normal.bound_ratio
        y7_Normal = res_Normal.max_angle  * pi / 180
        # y8_Normal = res_Normal.upper_error_bound
        x9_Normal = range(1, 2*res_Normal.outer_iters)
        #println("length x9 = ", length(x9_Normal))
        y9_Normal = res_Normal.norm_grad
        #println("length y9 = ", length(y9_Normal))

        res_NM = resvec_NM[j]
        x_NM  = range(1, res_NM.outer_iters)
        y1_NM = res_NM.als_error
        y2_NM = res_NM.inner_iters
        y3_NM = res_NM.time_cg
        y4_NM = res_NM.cg_last_resnorm_1
        y5_NM = res_NM.cg_last_resnorm_2
        
        # y6_NM = res_NM.bound_ratio
        y7_NM = res_NM.max_angle  * pi / 180
        # y8_NM = res_NM.upper_error_bound
        x9_NM = range(1, 2*res_NM.outer_iters)
        y9_NM = res_NM.norm_grad

        plot!(p[idx(j,1)], x_Normal, y1_Normal; label="ALS Normal", yscale=:log10, legend=false,
              xlabel="Outer iteration", title="ALS error: || X - B_r ||_F at each outer iteration($(names[j]))")
        # plot!(p[idx(j,1)], x_Normal, y8_Normal;label="Upper error bound in opnorm-2")
        plot!(p[idx(j,1)], x_NM, y1_NM;label="ALS Newton")

        # Compute error ratio 
        ratio_range_Normal = range(2,length(y1_Normal))
        ratio_Normal = [y1_Normal[k]/y1_Normal[k-1] for k in ratio_range_Normal]
        if !isempty(ratio_Normal)
            last_ratio_Normal[j] = ratio_Normal[end]
        else
            last_ratio_Normal[j] = NaN
        end
        # print("length(ratio) = ", length(ratio), "\n")
        xratio_Normal = x_Normal[2:end]
        plot!(p[idx(j,2)], xratio_Normal, ratio_Normal; label="ALS Normal", xlabel="Outer iteration",title="Error Ratio ($(names[j]))", legend=false)
        # plot!(p[idx(j,2)], x_Normal, y6_Normal;label="ratio of upper theoretical bound")

        ratio_range_NM = range(2,length(y1_NM))
        ratio_NM = [y1_NM[k]/y1_NM[k-1] for k in ratio_range_NM]
        if !isempty(ratio_NM)
            last_ratio_NM[j] = ratio_NM[end]
        else
            last_ratio_NM[j] = NaN
        end
        xratio_NM = x_NM[2:end]
        plot!(p[idx(j,2)], xratio_NM, ratio_NM;label="ALS Newton")

        plot!(p[idx(j,3)], x_Normal, y2_Normal; label="ALS Normal",legend=false,
        xlabel="Outer iteration", title="Inner iters ($(names[j]))")
        plot!(p[idx(j,3)], x_NM, y2_NM; label="ALS Newton")

        plot!(p[idx(j,4)], x_Normal, y3_Normal; label="ALS Normal",legend=false,
                xlabel="Outer iteration", title="Time ($(names[j]))")
        plot!(p[idx(j,4)], x_NM, y3_NM; label="ALS Newton")

        plot!(p[idx(j,5)], x_Normal, y4_Normal;label="left ALS Normal",
              xlabel="Outer iteration", title="left CG residual matrix in frobenius norm ($(names[j]))")
        plot!(p[idx(j,5)], x_NM, y4_NM;label="left ALS Newton")

        plot!(p[idx(j,6)], x_Normal, y5_Normal;label="right ALS Normal",
              xlabel="Outer iteration", title="right CG residual matrix in frobenius norm ($(names[j]))")
        plot!(p[idx(j,6)], x_NM, y5_NM;label="right ALS Newton")

        plot!(p[idx(j,7)], x_Normal, y7_Normal; legend=false, label="ALS Normal",
              xlabel="Outer iteration", title="Max principal angle between subspaces in degree ($(names[j]))")
        plot!(p[idx(j,7)], x_NM, y7_NM; legend=false, label="ALS Newton")


        plot!(p[idx(j,8)], x9_Normal, y9_Normal; legend=false, label="ALS Normal",
              xlabel="Outer iteration", title="Norm of gradient ($(names[j])) per full and half step")
        plot!(p[idx(j,8)], x9_NM, y9_NM; legend=false, label="ALS NM")

        x10_Normal = range(1, res_Normal.outer_iters)
        y10_Normal = [y9_Normal[2*s] for s in x10_Normal]
        x10_NM = range(1, res_NM.outer_iters)
        y10_NM = [y9_NM[2*s] for s in x10_NM]
    
        # Full step
        plot!(p[idx(j,9)], x10_Normal, y10_Normal; legend=false,label="Normal",yscale=:log10,
          xlabel="Outer iteration", title="ALS Normal and ALS Newton: Norm of gradient ($(names[j])) per full step")
        plot!(p[idx(j,9)], x10_NM, y10_NM; legend=false,label="Newton")


        error_mat_sol_Normal_Newton .= res_Normal.X1_sol * (res_Normal.X2_sol') - res_NM.X1_sol * (res_NM.X2_sol')
        error_norm_sol_Normal_Newton[j] = norm(error_mat_sol_Normal_Newton, 2)

        error_mat_X1_sol_Normal_Newton .= res_Normal.X1_sol - res_NM.X1_sol
        error_norm_X1_sol_Normal_Newton[j] = norm(error_mat_X1_sol_Normal_Newton, 2)

        error_mat_X2_sol_Normal_Newton .= res_Normal.X2_sol - res_NM.X2_sol
        error_norm_X2_sol_Normal_Newton[j] = norm(error_mat_X2_sol_Normal_Newton, 2)
        
    end

    if 1 ≤ r < length(singular_values)
            Sigma_r_square_ratio = (singular_values[r+1] / singular_values[r])^2
            Sigma_r_ratio = singular_values[r+1] / singular_values[r]
            singular_values_r = singular_values[r]
            singular_values_r1 = singular_values[r+1]
    else
            Sigma_r_square_ratio = NaN
            Sigma_r_ratio = NaN
            singular_values_r = singular_values[r]
            singular_values_r1 = NaN
    end

    num_threads = Threads.nthreads()
    headers = ["Num of threads","X size","rank","outer_tol", "inner_abstol", "inner_reltol","final_error: Normal | Newton", 
                "total_time: Normal | Newton", "total_inner_iters: Normal | Newton", "outer_iters: Normal | Newton", "converged: Normal | Newton", "final_ratio: Normal | Newton", 
                #"X2tX2_fnorm", 
                #"last_theoretical_error_bound", "last_theta_max", "last_theoretical_ratio", 
                "sigma_r", "sigma_{r+1}", "sigma_{r+1}/sigma_r", "(sigma_{r+1}/sigma_r)^2",
                "|| X_Normal - X_Newton ||_F",
                "|| X1_Normal - X1_Newton ||_F",
                "|| X2_Normal - X2_Newton ||_F"]
    rows = [(
        string(num_threads),
        string(size_B),
        string(resvec_Normal[j].rank),
        @sprintf("Normal = %.1e | Newton = %.1e", resvec_Normal[j].outer_tol, resvec_NM[j].outer_tol),
        @sprintf("Normal = %.1e | Newton = %.1e", resvec_Normal[j].inner_abstol, resvec_NM[j].inner_abstol),
        @sprintf("Normal = %.1e | Newton = %.1e", resvec_Normal[j].inner_reltol, resvec_NM[j].inner_reltol),
        @sprintf("Normal = %.3e | Newton = %.3e", last(resvec_Normal[j].als_error), last(resvec_NM[j].als_error)),
        @sprintf("Normal = %.3e | Newton = %.3e", resvec_Normal[j].total_time_cg, resvec_NM[j].total_time_cg),
        string("Normal = ", resvec_Normal[j].total_inner_iters, "| Newton = ", resvec_NM[j].total_inner_iters),
        string("Normal = ", resvec_Normal[j].outer_iters, "| Newton = ", resvec_NM[j].outer_iters),
        string("Normal = ", resvec_Normal[j].converged_als, "| Newton = ", resvec_NM[j].converged_als),
        @sprintf("Normal = %.4f | Newton = %.4f", last_ratio_Normal[j], last_ratio_NM[j]),
        #@sprintf("%.3e", resvec_Normal[j].X2tX2_fnorm),
        # @sprintf("%.3e", resvec_Normal[j].last_upper_error_bound),
        # @sprintf("%.3f", resvec_Normal[j].last_theta_max),
        # @sprintf("%.4e", resvec_Normal[j].last_theor_ratio),
        @sprintf("%.4e", singular_values_r),
        @sprintf("%.4e", singular_values_r1),
        @sprintf("%.4e", Sigma_r_ratio),
        @sprintf("%.4e", Sigma_r_square_ratio),
        @sprintf("%.4e", error_norm_sol_Normal_Newton[j]),
        @sprintf("%.4e", error_norm_X1_sol_Normal_Newton[j]),
        @sprintf("%.4e", error_norm_X2_sol_Normal_Newton[j])
    ) for j in 1:K]

    tplot = Kcol*K + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        nrows = length(rows) + 1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in range(1, length(rows))
            vals = rows[rrow]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1], text(vals[c], 16, :black, :center))
            end
        end
        annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end
    rows, cols = size_B
    matstr = "$(rows)x$(cols)"
    condstr = @sprintf("cond%.2f", cond_B)
    rankstr = "rank_$(r)"

    dir = "data/results_html/$(cg_method)/ALS_Normal_vs_ALS_Newton/$(matstr)_$(condstr)/$(rankstr)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(outfile)_als_rank_$(r).html"
    savefig(p, outfile_html)

    return outfile_html
end


function compare_2d_Id_als_NM_exact_and_NM_inexact(r::Int, resvec_NMExact::Vector{ALSResult}, 
                             resvec_NMInexact::Vector{ALSResult}, 
                             svd_ratios::Vector{Float64},singular_values::Vector{Float64},
                             size_B::Tuple{Int, Int}, cond_B::Float64;
                             cg_method = "IterativeSolvers_cg",
                             names = ["outer=$(resvec_NMExact[i].outer_tol), inner=$(resvec_NMExact[i].inner_abstol)" for i in range(1, length(resvec_NMExact))],
                             outfile = "ALS_Newton_Exact_and_ALS_Newton_Inexact")

    K = length(resvec_NMExact)
    Kcol = 9

    lay = @layout [grid(K, Kcol); T{0.1h}]
    p = plot(layout=lay, size=(10000, K*700+1000), margin=20mm, left_margin=30mm, 
             plot_title="ALS Newton Rank Panel: ALS Newton Exact and ALS Newton Inexact| matrix size = $(size_B) cond = $cond_B |  rank = $r |  method = $cg_method ")

    idx(row, col) = (row-1)*Kcol + col
    last_ratio_NMExact = zeros(K)
    last_ratio_NMInexact = zeros(K)
    error_norm_sol_NMExact_NMInexact = zeros(K)
    error_norm_X1_sol_NMExact_NMInexact = zeros(K)
    error_norm_X2_sol_NMExact_NMInexact = zeros(K)
    error_mat_sol_NMExact_NMInexact = zeros(size_B)
    error_mat_X1_sol_NMExact_NMInexact = zeros(size_B[1], r)
    error_mat_X2_sol_NMExact_NMInexact = zeros(size_B[2], r)

    for j in 1:K
        res_NMExact = resvec_NMExact[j]
        x_NMExact  = range(1, res_NMExact.outer_iters)
        y1_NMExact = res_NMExact.als_error
        y2_NMExact = res_NMExact.inner_iters
        y3_NMExact = res_NMExact.time_cg
        y4_NMExact = res_NMExact.cg_last_resnorm_1
        y5_NMExact = res_NMExact.cg_last_resnorm_2
        
        # y6_Normal = res_Normal.bound_ratio
        y7_NMExact = res_NMExact.max_angle  * pi / 180
        # y8_Normal = res_Normal.upper_error_bound

        x9_NMExact = range(1, 2*res_NMExact.outer_iters)
        y9_NMExact = res_NMExact.norm_grad

        res_NMInexact = resvec_NMInexact[j]
        x_NMInexact  = range(1, res_NMInexact.outer_iters)
        y1_NMInexact = res_NMInexact.als_error
        y2_NMInexact = res_NMInexact.inner_iters
        y3_NMInexact = res_NMInexact.time_cg
        y4_NMInexact = res_NMInexact.cg_last_resnorm_1
        y5_NMInexact = res_NMInexact.cg_last_resnorm_2
        
        # y6_NM = res_NM.bound_ratio
        y7_NMInexact = res_NMInexact.max_angle  * pi / 180
        # y8_NM = res_NM.upper_error_bound

        x9_NMInexact = range(1, 2*res_NMInexact.outer_iters)
        y9_NMInexact = res_NMInexact.norm_grad

        plot!(p[idx(j,1)], x_NMExact, y1_NMExact; label="ALS NMExact", yscale=:log10, legend=false,
              xlabel="Outer iteration", title="ALS error: || X - B_r ||_F at each outer iteration($(names[j]))")
        # plot!(p[idx(j,1)], x_Normal, y8_Normal;label="Upper error bound in opnorm-2")
        plot!(p[idx(j,1)], x_NMInexact, y1_NMInexact;label="ALS NMInexact")

        # Compute error ratio 
        ratio_NMExact = [y1_NMExact[k]/y1_NMExact[k-1] for k in range(2,length(y1_NMExact))]
        if !isempty(ratio_NMExact)
            last_ratio_NMExact[j] = ratio_NMExact[end]
        else
            last_ratio_NMExact[j] = NaN
        end
        # print("length(ratio) = ", length(ratio), "\n")
        plot!(p[idx(j,2)], x_NMExact[2:end], ratio_NMExact; label="ALS NMExact", xlabel="Outer iteration",title="Error Ratio ($(names[j]))", legend=false)
        # plot!(p[idx(j,2)], x_Normal, y6_Normal;label="ratio of upper theoretical bound")

        ratio_NMInexact = [y1_NMInexact[k]/y1_NMInexact[k-1] for k in range(2,length(y1_NMInexact))]
        if !isempty(ratio_NMInexact)
            last_ratio_NMInexact[j] = ratio_NMInexact[end]
        else
            last_ratio_NMInexact[j] = NaN
        end
        plot!(p[idx(j,2)], x_NMInexact[2:end], ratio_NMInexact;label="NMInexact")

        plot!(p[idx(j,3)], x_NMExact, y2_NMExact; label="ALS NMExact",legend=false,
        xlabel="Outer iteration", title="Inner iters ($(names[j]))")
        plot!(p[idx(j,3)], x_NMInexact, y2_NMInexact; label="NMInexact")

        plot!(p[idx(j,4)], x_NMExact, y3_NMExact; label="ALS NMExact",legend=false,
                xlabel="Outer iteration", title="Time ($(names[j]))")
        plot!(p[idx(j,4)], x_NMInexact, y3_NMInexact; label="NMInexact")


        plot!(p[idx(j,7)], x_NMExact, y7_NMExact; legend=false, label="ALS NMExact",
              xlabel="Outer iteration", title="Max principal angle between subspaces in degree ($(names[j]))")
        plot!(p[idx(j,7)], x_NMInexact, y7_NMInexact; legend=false, label="ALS NMInexact")

        
        plot!(p[idx(j,8)], x9_NMExact, y9_NMExact; legend=false, label="ALS NMExact",
              xlabel="Outer iteration", title="Norm of gradient ($(names[j])) per half and full step")
        plot!(p[idx(j,8)], x9_NMInexact, y9_NMInexact; legend=false, label="ALS NMInexact")

        x10_NMExact = range(1, res_NMExact.outer_iters)
        y10_NMExact= [y9_NMExact[2*s] for s in x10_NMExact]
        x10_NMInexact = range(1, res_NMInexact.outer_iters)
        y10_NMInexact = [y9_NMInexact[2*s] for s in x10_NMInexact]
    
        # Full step
        plot!(p[idx(j,9)], x10_NMExact, y10_NMExact; legend=false,label="Exact",yscale=:log10,
          xlabel="Outer iteration", title="ALS Newton exact and ALS Newton inexact: Norm of gradient ($(names[j])) per full step")
        plot!(p[idx(j,9)], x10_NMInexact, y10_NMInexact; legend=false,label="Inexact")

        error_mat_sol_NMExact_NMInexact .= res_NMExact.X1_sol * (res_NMExact.X2_sol') - res_NMInexact.X1_sol * (res_NMInexact.X2_sol')
        error_norm_sol_NMExact_NMInexact[j] = norm(error_mat_sol_NMExact_NMInexact, 2)

        error_mat_X1_sol_NMExact_NMInexact .= res_NMExact.X1_sol - res_NMInexact.X1_sol
        error_norm_X1_sol_NMExact_NMInexact[j] = norm(error_mat_X1_sol_NMExact_NMInexact, 2)

        error_mat_X2_sol_NMExact_NMInexact .= res_NMExact.X2_sol - res_NMInexact.X2_sol
        error_norm_X2_sol_NMExact_NMInexact[j] = norm(error_mat_X2_sol_NMExact_NMInexact, 2)
        
    end

    if 1 ≤ r < length(singular_values)
            Sigma_r_square_ratio = (singular_values[r+1] / singular_values[r])^2
            Sigma_r_ratio = singular_values[r+1] / singular_values[r]
            singular_values_r = singular_values[r]
            singular_values_r1 = singular_values[r+1]
    else
            Sigma_r_square_ratio = NaN
            Sigma_r_ratio = NaN
            singular_values_r = singular_values[r]
            singular_values_r1 = NaN
    end

    num_threads = Threads.nthreads()
    headers = ["Num of threads","X size","rank","outer_tol", "inner_abstol | inexact_coeff", "inner_reltol","final_error: NMExact | NMInexact", 
                "total_time: NMExact | NMInexact", "total_inner_iters: NMExactl | NMInexact", "outer_iters: NMExact | NMInexact", "converged: NMExact | NMInexactn", "final_ratio: NMExact | NMInexact", 
                #"X2tX2_fnorm", 
                #"last_theoretical_error_bound", "last_theta_max", "last_theoretical_ratio", 
                "sigma_r", "sigma_{r+1}", "sigma_{r+1}/sigma_r", "(sigma_{r+1}/sigma_r)^2",
                "|| X_NMExact - X_NMInexact ||_F",
                "|| X1_NMExact - X1_NMInexact ||_F",
                "|| X2_NMExact - X2_NMInexact ||_F"]
    rows = [(
        string(num_threads),
        string(size_B),
        string(resvec_NMExact[j].rank),
        @sprintf("NMExact = %.1e | NMInexact = %.1e", resvec_NMExact[j].outer_tol, resvec_NMInexact[j].outer_tol),
        @sprintf("NMExact abstol= %.1e | NMInexact coeff= %.1e", resvec_NMExact[j].inner_abstol, resvec_NMInexact[j].inexact_coeff),
        @sprintf("NMExact = %.1e | NMInexact = %.1e", resvec_NMExact[j].inner_reltol, resvec_NMInexact[j].inner_reltol),
        @sprintf("NMExact = %.3e | NMInexact = %.3e", last(resvec_NMExact[j].als_error), last(resvec_NMInexact[j].als_error)),
        @sprintf("NMExact = %.3e | NMInexact = %.3e", resvec_NMExact[j].total_time_cg, resvec_NMInexact[j].total_time_cg),
        string("NMExact = ", resvec_NMExact[j].total_inner_iters, "| NMInexact = ", resvec_NMInexact[j].total_inner_iters),
        string("NMExact = ", resvec_NMExact[j].outer_iters, "| NMInexact = ", resvec_NMInexact[j].outer_iters),
        string("NMExact = ", resvec_NMExact[j].converged_als, "| NMInexact = ", resvec_NMInexact[j].converged_als),
        @sprintf("NMExact = %.4f | NMInexact = %.4f", last_ratio_NMExact[j], last_ratio_NMInexact[j]),
        #@sprintf("%.3e", resvec_Normal[j].X2tX2_fnorm),
        # @sprintf("%.3e", resvec_Normal[j].last_upper_error_bound),
        # @sprintf("%.3f", resvec_Normal[j].last_theta_max),
        # @sprintf("%.4e", resvec_Normal[j].last_theor_ratio),
        @sprintf("%.4e", singular_values_r),
        @sprintf("%.4e", singular_values_r1),
        @sprintf("%.4e", Sigma_r_ratio),
        @sprintf("%.4e", Sigma_r_square_ratio),
        @sprintf("%.4e", error_norm_sol_NMExact_NMInexact[j]),
        @sprintf("%.4e", error_norm_X1_sol_NMExact_NMInexact[j]),
        @sprintf("%.4e", error_norm_X2_sol_NMExact_NMInexact[j])
    ) for j in 1:K]

    tplot = Kcol*K + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        nrows = length(rows) + 1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in range(1, length(rows))
            vals = rows[rrow]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1], text(vals[c], 16, :black, :center))
            end
        end
        annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end
    rows, cols = size_B
    matstr = "$(rows)x$(cols)"
    condstr = @sprintf("cond%.2f", cond_B)
    rankstr = "rank_$(r)"

    dir = "data/results_html/$(cg_method)/ALS_NMExact_vs_ALS_NMInexact/$(matstr)_$(condstr)/$(rankstr)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(outfile)_als_rank_$(r).html"
    savefig(p, outfile_html)

    return outfile_html
end