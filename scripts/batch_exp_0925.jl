using JLD2
using Plots
using Printf
using Measures
include("../src/als_0925.jl")
include("../src/plots_0926.jl")
plotlyjs()

# data = load("data/matrix/10x10/10x10_cond9.24.jld2")
# B = data["B"]
# cond_B = data["cond_B"]
# ranks = [3, 5, 7, 10]
# ranks=[3]

println("Detected CPU threads available on system: ", Sys.CPU_THREADS)
println("Julia is currently using threads: ", Threads.nthreads())

num_threads = Threads.nthreads()

matrix_dirs = readdir(joinpath(@__DIR__, "../data/new_matrix"), join=true)
println("successful read matrix dirs: ", matrix_dirs)

for dir in matrix_dirs
    if isdir(dir)
        matrix_files = filter(f -> endswith(f, ".jld2"), readdir(dir, join=true))
        println("matrix_files: ", matrix_files)

        for file in matrix_files
            local B 
            local cond_B
            @load file B cond_B

            n, m = size(B)
            ranks =[ceil(Int, 0.25*m), ceil(Int, 0.5*m), ceil(Int, 0.75*m), m]
            exact_results_by_rank = Dict{Int, Vector{ALSResult}}()
            results_by_inner_tol = Dict{Float64, Vector{ALSResult}}()
            tol_num = 13
            # tol_num = 1

            for r in ranks
                B_svdt, svd_error, svd_ratios, singular_values = svd_truncated(B, r)
                save_singular_values_csv(r, singular_values, svd_ratios, size(B), cond_B)
                resvec = Vector{ALSResult}(undef, tol_num)
                for k in range(1, tol_num)
                    outer_tol = 10.0^(-13 + k) 
                    resvec_inner = Vector{ALSResult}(undef, tol_num+1-k)
                    vec_inner_tol = Vector{Float64}(undef, tol_num+1-k)
                    vec_outer_iters = Vector{Int}(undef, tol_num+1-k)
                    for j in range(1, tol_num+1-k)
                        inner_tol = 10.0^(-13 + k+j-1) 
                        vec_inner_tol[j] = inner_tol
                        res = als_2d_qr(B, r, B_svdt;
                                    outer_tol=outer_tol, inner_tol=inner_tol)
                        resvec_inner[j] = res
                        vec_outer_iters[j] = res.outer_iters
                        if j == 1
                            resvec[k] = res
                        end
                    end
                    results_by_inner_tol[outer_tol] = resvec_inner
                    save_relaxation_panel_pdf(r, resvec_inner, svd_ratios, singular_values, outer_tol, size(B), cond_B; outfile="QR_comparison_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")

                    outertolstr = @sprintf("outertol_%1.0e", outer_tol)
                    save_rank_panel_pdf(r, resvec_inner, svd_ratios, singular_values, size(B), cond_B; outfile="$(outertolstr)/QR_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")
                end
                exact_results_by_rank[r] = resvec
                save_rank_panel_pdf(r, resvec,svd_ratios, singular_values, size(B), cond_B; outfile="QR_exact")
            end

            rows, cols = size(B)
            matstr  = "$(rows)x$(cols)"
            condstr = @sprintf("cond%.2f", cond_B)
            dir = "data/results_snapshot/$(matstr)_$(condstr)"
            mkpath(dir)

            snap = joinpath(dir, "QR_als_results_snapshot.jld2")
            jldopen(snap, "w") do f
                write(f, "exact_results_by_rank", exact_results_by_rank)
                write(f, "results_by_inner_tol",  results_by_inner_tol)
            end

            println("Saved snapshot to: ", snap)
        end
    end
end