using JLD2
using Printf

include("../src/als_IterativeSolvers.jl")
include("../src/plots.jl")
plotlyjs()

# data = load("data/matrix/10x10/10x10_cond9.24.jld2")
# B = data["B"]
# cond_B = data["cond_B"]
# ranks = [3, 5, 8, 10]
# ranks=[3]

println("Detected CPU threads available on system: ", Sys.CPU_THREADS)
println("Julia is currently using threads: ", Threads.nthreads())

num_threads = Threads.nthreads()

# matrix_dirs = readdir(joinpath(@__DIR__, "../data/matrix"), join=true)
matrix_dirs = readdir(joinpath(@__DIR__, "../data/test_matrix"), join=true)
# matrix_dirs = readdir(joinpath(@__DIR__, "../data/matrix_by_singular_values"), join=true)
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
            #ranks =[ceil(Int, 0.25*m), ceil(Int, 0.5*m), ceil(Int, 0.75*m), m]
            ranks = [5]
            # ranks =[ceil(Int, 0.25*m), ceil(Int, 0.5*m), ceil(Int, 0.75*m)]
            # exact_results_by_rank = Dict{Int, Vector{ALSResult}}() # store data only
            # results_by_inner_tol = Dict{Float64, Vector{ALSResult}}() # store data only
            tol_num = 5
            # tol_num = 1
            min_tol_outer = -13
            min_tol_inner = -13
            eta_inexact_coff = 10.0^(-8)

            for r in ranks
                B_svdt, svd_error, svd_ratios, singular_values, SigmaVt, square_sigma_ratio = svd_truncated(B, r)
                save_singular_values_csv(r, singular_values, svd_ratios, size(B), cond_B)
                resvec_NMExact = Vector{ALSResult}(undef, tol_num)
                resvec_NMInexact = Vector{ALSResult}(undef, tol_num)
                for k in range(1, tol_num)
                    outer_tol = 10.0^(min_tol_outer + k) 
                    resvec_NMExact_inner = Vector{ALSResult}(undef, tol_num+1-k)
                    resvec_NMInexact_inner = Vector{ALSResult}(undef, tol_num+1-k)
                    # vec_NMExact_inner_tol = Vector{Float64}(undef, tol_num+1-k)
                    # vec_NMInexact_inner_tol = Vector{Float64}(undef, tol_num+1-k)
                    vec_NMExact_outer_iters = Vector{Int}(undef, tol_num+1-k)
                    vec_NMInexact_outer_iters = Vector{Int}(undef, tol_num+1-k)
                    for j in range(1, tol_num+1-k)
                        NMExact_inner_tol = 10.0^(min_tol_inner + k+j-1) 
                        NMInexact_inner_tol = 10.0^(min_tol_inner + k+j-2) 
                        # vec_NMExact_inner_tol[j] = NMExact_inner_tol
                        # vec_NMInexact_inner_tol[j] = NMInexact_inner_tol
                        res_NMExact = als_2d_Id_newton(B, r, cond_B, square_sigma_ratio, B_svdt, SigmaVt;
                                                        outer_tol=outer_tol, inner_abstol=NMExact_inner_tol, inner_reltol=0.0)
                        res_NMInexact = als_2d_Id_inexact_newton(B, r, cond_B, square_sigma_ratio, B_svdt, SigmaVt,
                                                                eta_inexact_coff;
                                                                outer_tol=outer_tol, #inner_abstol=NMExact_inner_tol, 
                                                                inner_reltol=0.0)
                        resvec_NMExact_inner[j] = res_NMExact
                        vec_NMExact_outer_iters[j] = res_NMExact.outer_iters
                        resvec_NMInexact_inner[j] = res_NMInexact
                        vec_NMInexact_outer_iters[j] = res_NMInexact.outer_iters
                        if j == 1
                            resvec_NMExact[k] = res_NMExact
                            resvec_NMInexact[k] = res_NMInexact
                        end
                    end
                    # results_by_inner_tol[outer_tol] = resvec_inner
                    save_relaxation_panel_pdf(r, resvec_NMExact_inner, svd_ratios, singular_values, outer_tol, size(B), cond_B; cg_method = "IterativeSolvers_cg", als_method = "ALS_NMExact", outfile="ALS_NMExact_comparison_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")
                    save_relaxation_panel_pdf(r, resvec_NMInexact_inner, svd_ratios, singular_values, outer_tol, size(B), cond_B; cg_method = "IterativeSolvers_cg", als_method = "ALS_NMInexact", outfile="ALS_NMInexact_comparison_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")

                    outertolstr = @sprintf("outertol_%1.0e", outer_tol)
                    save_rank_panel_pdf(r, resvec_NMExact_inner, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", als_method = "ALS_NMExact",outfile="$(outertolstr)/ALS_NMExact_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")
                    save_rank_panel_pdf(r, resvec_NMInexact_inner, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", als_method = "ALS_NMInexact",outfile="$(outertolstr)/ALS_NMInexact_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")
                end
                # exact_results_by_rank[r] = resvec
                # save_rank_panel_pdf(r, resvec_normal, svd_ratios, singular_values, size(B), cond_B; cg_method = "ALS_Normal_IterativeSolvers_cg", outfile="ALS_Normal_exact")
                # save_rank_panel_pdf(r, resvec_NM, svd_ratios, singular_values, size(B), cond_B; cg_method = "ALS_NM_IterativeSolvers_cg", outfile="ALS_NM_exact")
                compare_2d_Id_als_NM_exact_and_NM_inexact(r, resvec_NMExact, 
                             resvec_NMInexact, 
                             svd_ratios,singular_values,
                             size(B), cond_B)
            end

            # rows, cols = size(B)
            # matstr  = "$(rows)x$(cols)"
            # condstr = @sprintf("cond%.2f", cond_B)
            # dir = "data/results_snapshot/IterativeSolvers_cg/$(matstr)_$(condstr)"
            # mkpath(dir)

            # snap = joinpath(dir, "QR_als_results_snapshot.jld2")
            # jldopen(snap, "w") do f
            #     write(f, "exact_results_by_rank", exact_results_by_rank)
            #     write(f, "results_by_inner_tol",  results_by_inner_tol)
            # end

            # println("Saved snapshot to: ", snap)
        end
    end
end

