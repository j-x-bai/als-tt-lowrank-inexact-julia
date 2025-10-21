using JLD2
using Plots
using Printf
using Measures
using Contour
plotlyjs()
# gr()


struct NewtonResult{T}
    vec_x::Vector{T}
    vec_y::Vector{T}              
    f_vals::Vector{T} 
    g_1::Vector{T} 
    g_2::Vector{T}
    delta_x::Vector{T}
    delta_y::Vector{T}      
    H11::Vector{T}            
    H12::Vector{T}
    H21::Vector{T}
    H22::Vector{T}
    f_change::Vector{T}  # |f_k - f_{k-1}|
    tol::T
    iters::Int
    converged::Bool
end


function NM_panel_pdf(Global_NMResults::Dict{Float64, NewtonResult}, 
                            ALS_NMResults::Dict{Float64, NewtonResult},
                            vec_epi::AbstractVector{<:Real},
                            f_vec::Dict{Float64, Function},
                            f_name_vec:: Dict{Float64, LaTeXString};
                            f_name="f4_sym",
                            vec_epi_name="1to100")

    length_epi = length(vec_epi)
    lay = @layout [grid(length_epi, 10); T{0.05h}]
    p = plot(layout=lay, size=(7000, length_epi*700), margin=20mm, left_margin=30mm, 
             plot_title="Global and ALS Newton Results by iterations | function = $(f_name) | vec_epi = $(vec_epi) ")

    idx(row, col) = (row-1)*10 + col
    

    for j in 1:length_epi
        epi = vec_epi[j]
        Global_res = Global_NMResults[epi]
        iters_G = Global_res.iters
        x_G  = range(1, iters_G+1) # + the initial point
        #z_G =  Global_res.z
        y1_G = Global_res.f_vals
        y2_G = Global_res.f_change
        y3_G = Global_res.g_1
        y4_G = Global_res.g_2
        y5_G = Global_res.H11
        y6_G = Global_res.H22
        y7_G = Global_res.H12
        y8_G = Global_res.H21
        y9_G = Global_res.delta_x
        y10_G = Global_res.delta_y

        ALS_res = ALS_NMResults[epi]
        iters_ALS = ALS_res.iters
        x_ALS  = range(1, iters_ALS+1) # + the initial point
        #z_ALS =  ALS_res.z
        y1_ALS = ALS_res.f_vals
        y2_ALS = ALS_res.f_change
        y3_ALS = ALS_res.g_1
        y4_ALS = ALS_res.g_2
        y5_ALS = ALS_res.H11
        y6_ALS = ALS_res.H22
        y7_ALS = ALS_res.H12
        y8_ALS = ALS_res.H21
        y9_ALS = ALS_res.delta_x
        y10_ALS = ALS_res.delta_y

        plot!(p[idx(j,1)], x_G, y1_G; label="Global Newton",
              xlabel="Iteration", title="f_vals when epi =$(epi)")
        plot!(p[idx(j,1)], x_ALS, y1_ALS;label="ALS Newton")

        plot!(p[idx(j,2)], x_G, y2_G;label="Global Newton",
               xlabel="Iteration", title="f_change when epi =$(epi)")
        plot!(p[idx(j,2)], x_ALS, y2_ALS;label="ALS Newton")

        plot!(p[idx(j,3)], x_G, y3_G;label="Global Newton",
             xlabel="Iteration",title="Grad_x when epi =$(epi)")
        plot!(p[idx(j,3)], x_ALS, y3_ALS;label="ALS Newton")

        plot!(p[idx(j,4)], x_G, y4_G; label="Global Newton",
             xlabel="Iteration", title="Grad_y when epi =$(epi)")
        plot!(p[idx(j,4)], x_ALS, y4_ALS;label="ALS Newton")

        plot!(p[idx(j,5)], x_G, y5_G; label="Global Newton",
            xlabel="Iteration", title="Hessian_xx when epi =$(epi)")
        plot!(p[idx(j,5)], x_ALS, y5_ALS;label="ALS Newton")

        plot!(p[idx(j,6)], x_G, y6_G; label="Global Newton",
            xlabel="Iteration", title="Hessian_yy when epi =$(epi)")
        plot!(p[idx(j,6)], x_ALS, y6_ALS;label="ALS Newton")

        plot!(p[idx(j,7)], x_G, y7_G;label="Global Newton",
            xlabel="Iteration", title="Hessian_xy when epi =$(epi)")
        plot!(p[idx(j,7)], x_ALS, y7_ALS;label="ALS Newton")

        plot!(p[idx(j,8)], x_G, y8_G;label="Global Newton",
            xlabel="Iteration", title="Hessian_yx when epi =$(epi)")
        plot!(p[idx(j,8)], x_ALS, y8_ALS;label="ALS Newton")

        plot!(p[idx(j,9)], x_G, y9_G;label="Global Newton",
            xlabel="Iteration", title="Delta_x when epi =$(epi)")
        plot!(p[idx(j,9)], x_ALS, y9_ALS;label="ALS Newton")

        plot!(p[idx(j,10)], x_G, y10_G;label="Global Newton",
            xlabel="Iteration", title="Delta_y when epi =$(epi)")
        plot!(p[idx(j,10)], x_ALS, y10_ALS;label="ALS Newton")
        
    end

    # headers = ["epi", "grad_tol = step_tol = frel_tol", "iters (Global)", "iters (NM)", "last f_vals (Global)", "last f_vals (NM)", "last solution (Global)", "last solution (NM)","converged (Global)", "converged (NM)","last error between Global and ALS"]
    headers = ["epi", "f", "grad_tol = step_tol = frel_tol", "iters (Global | NM)", "last f_vals (Global | NM)", "last solution x (Global | NM)", "last solution y (Global | NM)","converged (Global | NM)","last error between Global and ALS"]
    length_headers = length(headers)
    rows = Matrix{String}(undef, length_epi, length_headers)
    for j in 1:length_epi
            epi = vec_epi[j]
            Global_res = Global_NMResults[epi]
            ALS_res = ALS_NMResults[epi]
            rows[j,:] =[
            @sprintf("%.2e", epi),
            string(f_name_vec[epi]),
            @sprintf("%.2e", Global_res.tol),
            string("Global_NM = ", Global_res.iters, " ALS_NM = ", ALS_res.iters),
            @sprintf("%.3e| %.3e", Global_res.f_vals[end], ALS_res.f_vals[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_x[end], ALS_res.vec_x[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_y[end], ALS_res.vec_y[end]),
            string(Global_res.converged, ALS_res.converged),
            @sprintf("%.4f", abs(ALS_res.f_vals[end] - Global_res.f_vals[end]))]
    end

    tplot = 10*length_epi + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        #nrows = length(rows) + 1
        nrows = length_epi +1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in 1:size(rows, 1)
            vals = rows[rrow, :]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1],
                        text(vals[c], 16, :black, :center))
            end
        end
        #annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end

    dir = "data/results_html/$(f_name)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(vec_epi_name).html"
    savefig(p, outfile_html)

    return outfile_html
end



function NM_panel_pdf_full(Global_NMResults::Dict{Float64, NewtonResult}, 
                            ALS_NMResults::Dict{Float64, NewtonResult},
                            ALS_full_NMResults::Dict{Float64, NewtonResult},
                            vec_epi::AbstractVector{<:Real},
                            f_vec::Dict{Float64, Function},
                            f_name_vec:: Dict{Float64, LaTeXString};
                            f_name="f4_sym",
                            vec_epi_name="1to100_full")

    length_epi = length(vec_epi)
    lay = @layout [grid(length_epi, 10); T{0.05h}]
    p = plot(layout=lay, size=(7000, length_epi*700), margin=20mm, left_margin=30mm, 
             plot_title="Global and ALS Newton Results by iterations | function = $(f_name) | vec_epi = $(vec_epi) ")

    idx(row, col) = (row-1)*10 + col
    

    for j in 1:length_epi
        epi = vec_epi[j]
        Global_res = Global_NMResults[epi]
        iters_G = Global_res.iters
        x_G  = range(1, iters_G+1) # + the initial point
        y1_G = Global_res.f_vals
        y2_G = Global_res.f_change
        y3_G = Global_res.g_1
        y4_G = Global_res.g_2
        y5_G = Global_res.H11
        y6_G = Global_res.H22
        y7_G = Global_res.H12
        y8_G = Global_res.H21
        y9_G = Global_res.delta_x
        y10_G = Global_res.delta_y

        ALS_res = ALS_NMResults[epi]
        iters_ALS = ALS_res.iters
        x_ALS  = range(1, iters_ALS+1) # + the initial point
        y1_ALS = ALS_res.f_vals
        y2_ALS = ALS_res.f_change
        y3_ALS = ALS_res.g_1
        y4_ALS = ALS_res.g_2
        y5_ALS = ALS_res.H11
        y6_ALS = ALS_res.H22
        y7_ALS = ALS_res.H12
        y8_ALS = ALS_res.H21
        y9_ALS = ALS_res.delta_x
        y10_ALS = ALS_res.delta_y

        ALS_res_full = ALS_full_NMResults[epi]
        iters_ALS_full = ALS_res_full.iters
        x_ALS_full  = range(1, iters_ALS_full+1) # + the initial point
        x_ALS_scaled = x_ALS_full ./ 2 .+ 0.5
        y1_ALS_full = ALS_res_full.f_vals
        y2_ALS_full = ALS_res_full.f_change
        y3_ALS_full = ALS_res_full.g_1
        y4_ALS_full = ALS_res_full.g_2
        y5_ALS_full = ALS_res_full.H11
        y6_ALS_full = ALS_res_full.H22
        y7_ALS_full = ALS_res_full.H12
        y8_ALS_full = ALS_res_full.H21
        y9_ALS_full = ALS_res_full.delta_x
        y10_ALS_full = ALS_res_full.delta_y

        plot!(p[idx(j,1)], x_G, y1_G; label="Global Newton",
              xlabel="Iteration", title="f_vals when epi =$(epi)")
        plot!(p[idx(j,1)], x_ALS, y1_ALS;label="ALS Newton")
        plot!(p[idx(j,1)], x_ALS_scaled, y1_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,2)], x_G, y2_G;label="Global Newton",
               xlabel="Iteration", title="f_change when epi =$(epi)")
        plot!(p[idx(j,2)], x_ALS, y2_ALS;label="ALS Newton")
        plot!(p[idx(j,2)], x_ALS_scaled, y2_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,3)], x_G, y3_G;label="Global Newton",
             xlabel="Iteration",title="Grad_x when epi =$(epi)")
        plot!(p[idx(j,3)], x_ALS, y3_ALS;label="ALS Newton")
        plot!(p[idx(j,3)], x_ALS_scaled, y3_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,4)], x_G, y4_G; label="Global Newton",
             xlabel="Iteration", title="Grad_y when epi =$(epi)")
        plot!(p[idx(j,4)], x_ALS, y4_ALS;label="ALS Newton")
        plot!(p[idx(j,4)], x_ALS_scaled, y4_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,5)], x_G, y5_G; label="Global Newton",
            xlabel="Iteration", title="Hessian_xx when epi =$(epi)")
        plot!(p[idx(j,5)], x_ALS, y5_ALS;label="ALS Newton")
        plot!(p[idx(j,5)], x_ALS_scaled, y5_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,6)], x_G, y6_G; label="Global Newton",
            xlabel="Iteration", title="Hessian_yy when epi =$(epi)")
        plot!(p[idx(j,6)], x_ALS, y6_ALS;label="ALS Newton")
        plot!(p[idx(j,6)], x_ALS_scaled, y6_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,7)], x_G, y7_G;label="Global Newton",
            xlabel="Iteration", title="Hessian_xy when epi =$(epi)")
        plot!(p[idx(j,7)], x_ALS, y7_ALS;label="ALS Newton")
        plot!(p[idx(j,7)], x_ALS_scaled, y7_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,8)], x_G, y8_G;label="Global Newton",
            xlabel="Iteration", title="Hessian_yx when epi =$(epi)")
        plot!(p[idx(j,8)], x_ALS, y8_ALS;label="ALS Newton")
        plot!(p[idx(j,8)], x_ALS_scaled, y8_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,9)], x_G, y9_G;label="Global Newton",
            xlabel="Iteration", title="Delta_x when epi =$(epi)")
        plot!(p[idx(j,9)], x_ALS, y9_ALS;label="ALS Newton")
        plot!(p[idx(j,9)], x_ALS_scaled, y9_ALS_full;label="Full ALS Newton")

        plot!(p[idx(j,10)], x_G, y10_G;label="Global Newton",
            xlabel="Iteration", title="Delta_y when epi =$(epi)")
        plot!(p[idx(j,10)], x_ALS, y10_ALS;label="ALS Newton")
        plot!(p[idx(j,10)], x_ALS_scaled, y10_ALS_full;label="Full ALS Newton")
        
    end

    # headers = ["epi", "grad_tol = step_tol = frel_tol", "iters (Global)", "iters (NM)", "last f_vals (Global)", "last f_vals (NM)", "last solution (Global)", "last solution (NM)","converged (Global)", "converged (NM)","last error between Global and ALS"]
    headers = ["epi", "f", "grad_tol = step_tol = frel_tol", "iters (Global | NM)", "last f_vals (Global | NM)", "last solution x (Global | NM)", "last solution y (Global | NM)","converged (Global | NM)","last error between Global and ALS"]
    length_headers = length(headers)
    rows = Matrix{String}(undef, length_epi, length_headers)
    for j in 1:length_epi
            epi = vec_epi[j]
            Global_res = Global_NMResults[epi]
            ALS_res = ALS_NMResults[epi]
            rows[j,:] =[
            @sprintf("%.2e", epi),
            string(f_name_vec[epi]),
            @sprintf("%.2e", Global_res.tol),
            string("Global_NM = ", Global_res.iters, " ALS_NM = ", ALS_res.iters),
            @sprintf("%.3e| %.3e", Global_res.f_vals[end], ALS_res.f_vals[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_x[end], ALS_res.vec_x[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_y[end], ALS_res.vec_y[end]),
            string(Global_res.converged, ALS_res.converged),
            @sprintf("%.4f", abs(ALS_res.f_vals[end] - Global_res.f_vals[end]))]
    end

    tplot = 10*length_epi + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        #nrows = length(rows) + 1
        nrows = length_epi +1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in 1:size(rows, 1)
            vals = rows[rrow, :]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1],
                        text(vals[c], 16, :black, :center))
            end
        end
        #annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end

    dir = "data/results_html/$(f_name)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(vec_epi_name)_full.html"
    savefig(p, outfile_html)

    return outfile_html
end



function NM_panel_pdf_log(Global_NMResults::Dict{Float64, NewtonResult}, 
                            ALS_NMResults::Dict{Float64, NewtonResult},
                            vec_epi::AbstractVector{<:Real},
                            f_vec::Dict{Float64, Function},
                            f_name_vec:: Dict{Float64, LaTeXString};
                            f_name="f4_sym",
                            vec_epi_name="1to100")

    length_epi = length(vec_epi)
    lay = @layout [grid(length_epi, 10); T{0.05h}]
    p = plot(layout=lay, size=(7000, length_epi*700), margin=20mm, left_margin=30mm, 
             plot_title="Log10---Global and ALS Newton Results by iterations | function = $(f_name) | vec_epi = $(vec_epi) ")

    idx(row, col) = (row-1)*10 + col
    

    for j in 1:length_epi
        epi = vec_epi[j]
        Global_res = Global_NMResults[epi]
        iters_G = Global_res.iters
        x_G  = range(1, iters_G+1) # + the initial point
        #z_G =  Global_res.z
        y1_G = Global_res.f_vals
        y2_G = Global_res.f_change
        y3_G = Global_res.g_1
        y4_G = Global_res.g_2
        y5_G = Global_res.H11
        y6_G = Global_res.H22
        y7_G = Global_res.H12
        y8_G = Global_res.H21
        y9_G = Global_res.delta_x
        y10_G = Global_res.delta_y

        ALS_res = ALS_NMResults[epi]
        iters_ALS = ALS_res.iters
        x_ALS  = range(1, iters_ALS+1) # + the initial point
        #z_ALS =  ALS_res.z
        y1_ALS = ALS_res.f_vals
        y2_ALS = ALS_res.f_change
        y3_ALS = ALS_res.g_1
        y4_ALS = ALS_res.g_2
        y5_ALS = ALS_res.H11
        y6_ALS = ALS_res.H22
        y7_ALS = ALS_res.H12
        y8_ALS = ALS_res.H21
        y9_ALS = ALS_res.delta_x
        y10_ALS = ALS_res.delta_y

        plot!(p[idx(j,1)], x_G, y1_G; label="Global Newton", yscale=:log10,
              xlabel="Iteration", title="f_vals when epi =$(epi)")
        plot!(p[idx(j,1)], x_ALS, y1_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,2)], x_G, y2_G;yscale=:log10,label="Global Newton",
               xlabel="Iteration", title="f_change when epi =$(epi)")
        plot!(p[idx(j,2)], x_ALS, y2_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,3)], x_G, y3_G;label="Global Newton",yscale=:log10,
             xlabel="Iteration",title="Grad_x when epi =$(epi)")
        plot!(p[idx(j,3)], x_ALS, y3_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,4)], x_G, y4_G; label="Global Newton",yscale=:log10,
             xlabel="Iteration", title="Grad_y when epi =$(epi)")
        plot!(p[idx(j,4)], x_ALS, y4_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,5)], x_G, y5_G; label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_xx when epi =$(epi)")
        plot!(p[idx(j,5)], x_ALS, y5_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,6)], x_G, y6_G; label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_yy when epi =$(epi)")
        plot!(p[idx(j,6)], x_ALS, y6_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,7)], x_G, y7_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_xy when epi =$(epi)")
        plot!(p[idx(j,7)], x_ALS, y7_ALS;lyscale=:log10,abel="ALS Newton")

        plot!(p[idx(j,8)], x_G, y8_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_yx when epi =$(epi)")
        plot!(p[idx(j,8)], x_ALS, y8_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,9)], x_G, y9_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Delta_x when epi =$(epi)")
        plot!(p[idx(j,9)], x_ALS, y9_ALS;yscale=:log10,label="ALS Newton")

        plot!(p[idx(j,10)], x_G, y10_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Delta_y when epi =$(epi)")
        plot!(p[idx(j,10)], x_ALS, y10_ALS;yscale=:log10,label="ALS Newton")
        
    end

    # headers = ["epi", "grad_tol = step_tol = frel_tol", "iters (Global)", "iters (NM)", "last f_vals (Global)", "last f_vals (NM)", "last solution (Global)", "last solution (NM)","converged (Global)", "converged (NM)","last error between Global and ALS"]
    headers = ["epi", "f", "grad_tol = step_tol = frel_tol", "iters (Global | NM)", "last f_vals (Global | NM)", "last solution x (Global | NM)", "last solution y (Global | NM)","converged (Global | NM)","last error between Global and ALS"]
    length_headers = length(headers)
    rows = Matrix{String}(undef, length_epi, length_headers)
    for j in 1:length_epi
            epi = vec_epi[j]
            Global_res = Global_NMResults[epi]
            ALS_res = ALS_NMResults[epi]
            rows[j,:] =[
            @sprintf("%.2e", epi),
            string(f_name_vec[epi]),
            @sprintf("%.2e", Global_res.tol),
            string("Global_NM = ", Global_res.iters, " ALS_NM = ", ALS_res.iters),
            @sprintf("%.3e| %.3e", Global_res.f_vals[end], ALS_res.f_vals[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_x[end], ALS_res.vec_x[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_y[end], ALS_res.vec_y[end]),
            string(Global_res.converged, ALS_res.converged),
            @sprintf("%.4f", abs(ALS_res.f_vals[end] - Global_res.f_vals[end]))]
    end

    tplot = 10*length_epi + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        #nrows = length(rows) + 1
        nrows = length_epi +1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in 1:size(rows, 1)
            vals = rows[rrow, :]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1],
                        text(vals[c], 16, :black, :center))
            end
        end
        #annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end

    dir = "data/results_html/$(f_name)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(vec_epi_name)_log.html"
    savefig(p, outfile_html)

    return outfile_html
end



function NM_panel_pdf_full_log(Global_NMResults::Dict{Float64, NewtonResult}, 
                            ALS_NMResults::Dict{Float64, NewtonResult},
                            ALS_full_NMResults::Dict{Float64, NewtonResult},
                            vec_epi::AbstractVector{<:Real},
                            f_vec::Dict{Float64, Function},
                            f_name_vec:: Dict{Float64, LaTeXString};
                            f_name="f4_sym",
                            vec_epi_name="1to100_full")

    length_epi = length(vec_epi)
    lay = @layout [grid(length_epi, 10); T{0.05h}]
    p = plot(layout=lay, size=(7000, length_epi*700), margin=20mm, left_margin=30mm, 
             plot_title="Log10---Global and ALS Newton Results by iterations | function = $(f_name) | vec_epi = $(vec_epi) ")

    idx(row, col) = (row-1)*10 + col
    

    for j in 1:length_epi
        epi = vec_epi[j]
        Global_res = Global_NMResults[epi]
        iters_G = Global_res.iters
        x_G  = range(1, iters_G+1) # + the initial point
        y1_G = Global_res.f_vals
        y2_G = Global_res.f_change
        y3_G = Global_res.g_1
        y4_G = Global_res.g_2
        y5_G = Global_res.H11
        y6_G = Global_res.H22
        y7_G = Global_res.H12
        y8_G = Global_res.H21
        y9_G = Global_res.delta_x
        y10_G = Global_res.delta_y

        ALS_res = ALS_NMResults[epi]
        iters_ALS = ALS_res.iters
        x_ALS  = range(1, iters_ALS+1) # + the initial point
        y1_ALS = ALS_res.f_vals
        y2_ALS = ALS_res.f_change
        y3_ALS = ALS_res.g_1
        y4_ALS = ALS_res.g_2
        y5_ALS = ALS_res.H11
        y6_ALS = ALS_res.H22
        y7_ALS = ALS_res.H12
        y8_ALS = ALS_res.H21
        y9_ALS = ALS_res.delta_x
        y10_ALS = ALS_res.delta_y

        ALS_res_full = ALS_full_NMResults[epi]
        iters_ALS_full = ALS_res_full.iters
        x_ALS_full  = range(1, iters_ALS_full+1) # + the initial point
        x_ALS_scaled = x_ALS_full ./ 2 .+ 0.5
        y1_ALS_full = ALS_res_full.f_vals
        y2_ALS_full = ALS_res_full.f_change
        y3_ALS_full = ALS_res_full.g_1
        y4_ALS_full = ALS_res_full.g_2
        y5_ALS_full = ALS_res_full.H11
        y6_ALS_full = ALS_res_full.H22
        y7_ALS_full = ALS_res_full.H12
        y8_ALS_full = ALS_res_full.H21
        y9_ALS_full = ALS_res_full.delta_x
        y10_ALS_full = ALS_res_full.delta_y

        plot!(p[idx(j,1)], x_G, y1_G; label="Global Newton",yscale=:log10,
              xlabel="Iteration", title="f_vals when epi =$(epi)")
        plot!(p[idx(j,1)], x_ALS, y1_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,1)], x_ALS_scaled, y1_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,2)], x_G, y2_G;label="Global Newton",yscale=:log10,
               xlabel="Iteration", title="f_change when epi =$(epi)")
        plot!(p[idx(j,2)], x_ALS, y2_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,2)], x_ALS_scaled, y2_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,3)], x_G, y3_G;label="Global Newton",yscale=:log10,
             xlabel="Iteration",title="Grad_x when epi =$(epi)")
        plot!(p[idx(j,3)], x_ALS, y3_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,3)], x_ALS_scaled, y3_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,4)], x_G, y4_G; label="Global Newton",yscale=:log10,
             xlabel="Iteration", title="Grad_y when epi =$(epi)")
        plot!(p[idx(j,4)], x_ALS, y4_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,4)], x_ALS_scaled, y4_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,5)], x_G, y5_G; label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_xx when epi =$(epi)")
        plot!(p[idx(j,5)], x_ALS, y5_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,5)], x_ALS_scaled, y5_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,6)], x_G, y6_G; label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_yy when epi =$(epi)")
        plot!(p[idx(j,6)], x_ALS, y6_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,6)], x_ALS_scaled, y6_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,7)], x_G, y7_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_xy when epi =$(epi)")
        plot!(p[idx(j,7)], x_ALS, y7_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,7)], x_ALS_scaled, y7_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,8)], x_G, y8_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Hessian_yx when epi =$(epi)")
        plot!(p[idx(j,8)], x_ALS, y8_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,8)], x_ALS_scaled, y8_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,9)], x_G, y9_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Delta_x when epi =$(epi)")
        plot!(p[idx(j,9)], x_ALS, y9_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,9)], x_ALS_scaled, y9_ALS_full;yscale=:log10,label="Full ALS Newton")

        plot!(p[idx(j,10)], x_G, y10_G;label="Global Newton",yscale=:log10,
            xlabel="Iteration", title="Delta_y when epi =$(epi)")
        plot!(p[idx(j,10)], x_ALS, y10_ALS;yscale=:log10,label="ALS Newton")
        plot!(p[idx(j,10)], x_ALS_scaled, y10_ALS_full;yscale=:log10,label="Full ALS Newton")
        
    end

    # headers = ["epi", "grad_tol = step_tol = frel_tol", "iters (Global)", "iters (NM)", "last f_vals (Global)", "last f_vals (NM)", "last solution (Global)", "last solution (NM)","converged (Global)", "converged (NM)","last error between Global and ALS"]
    headers = ["epi", "f", "grad_tol = step_tol = frel_tol", "iters (Global | NM)", "last f_vals (Global | NM)", "last solution x (Global | NM)", "last solution y (Global | NM)","converged (Global | NM)","last error between Global and ALS"]
    length_headers = length(headers)
    rows = Matrix{String}(undef, length_epi, length_headers)
    for j in 1:length_epi
            epi = vec_epi[j]
            Global_res = Global_NMResults[epi]
            ALS_res = ALS_NMResults[epi]
            rows[j,:] =[
            @sprintf("%.2e", epi),
            string(f_name_vec[epi]),
            @sprintf("%.2e", Global_res.tol),
            string("Global_NM = ", Global_res.iters, " ALS_NM = ", ALS_res.iters),
            @sprintf("%.3e| %.3e", Global_res.f_vals[end], ALS_res.f_vals[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_x[end], ALS_res.vec_x[end]),
            @sprintf("%.3e| %.3e", Global_res.vec_y[end], ALS_res.vec_y[end]),
            string(Global_res.converged, ALS_res.converged),
            @sprintf("%.4f", abs(ALS_res.f_vals[end] - Global_res.f_vals[end]))]
    end

    tplot = 10*length_epi + 1
    plot!(p[tplot], framestyle=:none, legend=false, xlim=(0,1), ylim=(0,1))
    ncols = length(headers)
    xgrid = range(0.02, 0.98; length=ncols)
    if isempty(rows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], 0.95, text(h, 16, :black, :center)) 
        end
        annotate!(p[tplot], 0.5, 0.5, text("No data available", 16, :center, :red))
    else
        #nrows = length(rows) + 1
        nrows = length_epi +1
        ygrid = range(0.90, 0.10; length=nrows)
        for (c, h) in enumerate(headers)
            annotate!(p[tplot], xgrid[c], ygrid[1], text(h, 16, :black, :center))
        end
        for rrow in 1:size(rows, 1)
            vals = rows[rrow, :]
            for c in 1:ncols
                annotate!(p[tplot], xgrid[c], ygrid[rrow+1],
                        text(vals[c], 16, :black, :center))
            end
        end
        #annotate!(p[tplot], 0.02, 0.98, text("Matrix size $(size_B) Condition number $(cond_B) Rank = $r  (summary of $K configs)", 11, :black, :left))
    end

    dir = "data/results_html/$(f_name)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(vec_epi_name)_full_log.html"
    savefig(p, outfile_html)

    return outfile_html
end




function plot_2d_solution(Global_NMResults::Dict{Float64, NewtonResult}, 
                            ALS_NMResults::Dict{Float64, NewtonResult},
                            vec_epi::AbstractVector{<:Real},
                            f_vec::Dict{Float64, Function},
                            f_name_vec:: Dict{Float64, LaTeXString};
                            f_name="f4_sym",
                            vec_epi_name="1to100")

    
    length_epi = length(vec_epi)

    num_col = 3

    lay = @layout [grid(length_epi, num_col)]
    p = plot(layout=lay, size=(length_epi*700, num_col*700), #margin=10mm, left_margin=20mm,
             plot_title="2d plots of f | function = $(f_name)| vec_epi = $(vec_epi) ")
    idx(row, col) = (row-1)*num_col + col

    xs = ys = range(-1, stop=6, length=100)
    # xs_grad = ys_grad = range(-2, stop=10, length=50) 
    # nx, ny = length(xs_grad), length(ys_grad)
    # U_bg = Array{Float64}(undef, nx, ny)
    # V_bg = Array{Float64}(undef, nx, ny)
    
    for j in 1:length_epi

        epi = vec_epi[j]
        f_j = f_vec[epi] 
        
        fs = [f_j([x, y]) for x in xs, y in ys]
    
        plot!(p[idx(j,1)], xlabel="x", ylabel="y", zlabel="f(x,y)",
            title="Surface and Contour for f when epi = $(vec_epi[j])")

        plot!(p[idx(j,1)], xs, ys, fs; 
              seriestype=:surface, fillalpha=0.6, colorbar=false, label="f")

        cs = contours(xs, ys, fs)
        for cl in levels(cs)
            lvl = level(cl)  
            for line in lines(cl)
                _xs, _ys = coordinates(line)
                _fs = fill(lvl, length(_xs)) 
                
                plot!(p[idx(j,1)], _xs, _ys, _fs; 
                      color=:black, alpha=0.5, lw=1, label="")
                
                plot!(p[idx(j,1)], _xs, _ys, zeros(length(_xs)); 
                      color=:gray, alpha=0.5, lw=1, ls=:dash, label="")
            end
        end
        
        Global_res = Global_NMResults[epi]
        ALS_res = ALS_NMResults[epi]
        x_G_path = Global_res.vec_x
        y_G_path = Global_res.vec_y
        x_ALS_path = ALS_res.vec_x
        y_ALS_path = ALS_res.vec_y


        # plot!(p[idx(j,2)], xs, ys, fs; seriestype=:contour, 
        #       xlabel="x", ylabel="y", aspect_ratio=:equal, 
        #       title="2D Contour and Paths (Global and ALS) when epi = $(vec_epi[j])", colorbar=false)

        # plot!(p[idx(j,2)], x_G_path, y_G_path; seriestype=:path, color=:red, lw=2, label="Global NM")
        # scatter!(p[idx(j,2)], [x_G_path[end]], [y_G_path[end]]; color=:red, marker=:x, label="Global End")
                 
        # plot!(p[idx(j,2)], x_ALS_path, y_ALS_path; seriestype=:path, color=:blue, lw=2, label="ALS NM")
        # scatter!(p[idx(j,2)], [x_ALS_path[end]], [y_ALS_path[end]]; color=:blue, marker=:x, label="ALS End")
        

        # plot!(p[idx(j,3)], xs, ys, fs; seriestype=:surface, fillalpha=0.5, 
        #       xlabel="x", ylabel="y", zlabel="f(x,y)", 
        #       title="3D Surface and Paths when epi = $(vec_epi[j])", colorbar=false)
              
        z_G_vals = Global_res.f_vals
        z_ALS_vals = ALS_res.f_vals

        plot!(p[idx(j,1)], x_G_path, y_G_path, z_G_vals; seriestype=:path, 
              color=:red, lw=3, label="Global NM Path")
              
        plot!(p[idx(j,1)], x_ALS_path, y_ALS_path, z_ALS_vals; seriestype=:path, 
              color=:blue, lw=3, label="ALS NM Path")


        plot!(p[idx(j,2)], xs, ys, fs; seriestype=:contourf, 
              xlabel="x", ylabel="y", aspect_ratio=:equal, 
              title="Contour for Global NM and ALS NM when epi = $(vec_epi[j])", colorbar=false)

        plot!(p[idx(j,2)], x_G_path, y_G_path; seriestype=:path, color=:red, lw=2, label="Global NM")
        scatter!(p[idx(j,2)], [x_G_path[end]], [y_G_path[end]]; color=:red, marker=:x, label="Global End")
        
        plot!(p[idx(j,2)], x_ALS_path, y_ALS_path; seriestype=:path, color=:blue, lw=2, label="ALS NM")
        scatter!(p[idx(j,2)], [x_ALS_path[end]], [y_ALS_path[end]]; color=:blue, marker=:x, label="ALS End")


    end

    dir = "data_ALS_NM/results_html/$(f_name)"
    mkpath(dir) 
    outfile_html = "$(dir)/$(vec_epi_name)_2d_plots.html"
    savefig(p, outfile_html)

    return outfile_html
end