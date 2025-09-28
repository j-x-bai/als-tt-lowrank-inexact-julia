using JLD2
using Plots
using Printf
using Measures
plotlyjs()


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
#     max_angle::Vector{Float64}
#     upper_error_bound::Vector{Float64}
#     last_theta_max::Float64
#     last_upper_error_bound::Float64
# end

function save_rank_panel_pdf(r::Int, resvec::Vector{ALSResult}, 
                             svd_ratios::Vector{Float64},singular_values::Vector{Float64},
                             size_B::Tuple{Int, Int}, cond_B::Float64;
                             names = ["outer=$(resvec[i].outer_tol), inner=$(resvec[i].inner_abstol)" for i in range(1, length(resvec))],
                             outfile = "exact")

    K = length(resvec)

    lay = @layout [grid(K, 6); T{0.05h}]
    p = plot(layout=lay, size=(7000, K*700), margin=20mm, left_margin=30mm, 
             plot_title="Exact ALS Rank Panel | matrix size = $(size_B) cond = $cond_B ")

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
              xlabel="Outer iteration", title="Original error in Frobenius norm vs cond_B * sin(max_theta) in 2-norm at each outer iteration($(names[j]))")
        plot!(p[idx(j,1)], x, y8;label="Upper error bound in opnorm-2")

        # Compute error ratio 
        # als_error[s] = norm(X1 * X2' - B_svd_truncated, 2)
        # y1 = res.als_error
        ratio = [y1[k]/y1[k-1] for k in range(2,length(y1))]
        if !isempty(ratio)
            last_ratio[j] = ratio[end]
        else
            last_ratio[j] = NaN
        end
        print("length(ratio) = ", length(ratio), "\n")
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
        @sprintf("%.3f", resvec[j].X2tX2_fnorm),
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
    
    dir = "data/results_pdf/$(matstr)_$(condstr)/$(rankstr)"
    mkpath(dir) 
    for res in resvec
        outertolstr = @sprintf("outertol_%1.0e", res.outer_tol)
        outertoldir = "data/results_pdf/$(matstr)_$(condstr)/$(rankstr)/$(outertolstr)"
        mkpath(outertoldir)
    end
    outfile_pdf = "$(dir)/$(outfile)_als_rank_$(r).pdf"
    # savefig(p, outfile_pdf)
    outfile_html = "$(dir)/$(outfile)_als_rank_$(r).html"
    savefig(p, outfile_html)

    return outfile_pdf, outfile_html
end


function save_relaxation_panel_pdf(
    r::Int,
    resvec_inner::Vector{ALSResult},
    svd_ratios::Vector{Float64},
    singular_values::Vector{Float64},
    outer_tol::Float64,
    size_B::Tuple{Int, Int}, cond_B::Float64;
    outfile::AbstractString = "als_relaxation",
)

    lay = @layout [grid(7,1)]
    p = plot(layout=lay, size=(5000, 7000), margin=10mm, left_margin=20mm,
             plot_title="ALS Relaxation Panel |matrix size = $(size_B) cond = $cond_B |  rank = $r| outer_tol=$(@sprintf("%.1e", outer_tol)) ")

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
        string("%.3f", res.X2tX2_fnorm),
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
    dir = "data/results_pdf/$(matstr)_$(condstr)/$(rankstr)/$(outertolstr)"
    mkpath(dir)
    # outpath = "$(dir)/$(outfile)_als_rank_$(r).pdf"

    outpath_pdf = "$(dir)/$(outfile)_als_rank_$(r).pdf"
    # savefig(p, outpath_pdf)
    outpath_html = "$(dir)/$(outfile)_als_rank_$(r).html"
    savefig(p, outpath_html)
    return outpath_pdf, outpath_html
end


function save_singular_values_csv(r::Int, singular_values::Vector{Float64}, svd_ratios::Vector{Float64},
                                size_B::Tuple{Int, Int}, cond_B::Float64
                                ; outfile="singular_values_table")
    rows, cols = size_B
    matstr  = "$(rows)x$(cols)"
    condstr = @sprintf("cond%.2f", cond_B)
    rankstr = "rank_$(r)"
    dir = "data/results_pdf/$(matstr)_$(condstr)/$(rankstr)"
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