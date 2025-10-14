using JLD2
using Printf

include("../src/als_IterativeSolvers.jl")
include("../src/plots.jl")
plotlyjs()

println("Detected CPU threads available on system: ", Sys.CPU_THREADS)
println("Julia is currently using threads: ", Threads.nthreads())

num_threads = Threads.nthreads()

#----------------------------------------------------------------------------------------------------------------#
# Load test matrix
#data = load("data/test_matrix/10x10/10x10_cond9.24.jld2")
data = load("data/matrix_by_sigular_values/10x10/10x10_cond4.00.jld2")
B = data["B"]
n, m = size(B)
cond_B = data["cond_B"]
#----------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
# Tests for a fixed rank r and fixed outer tolrance
r = 7 # low-rank
k = 0 # fixed tolerance index: outer_tol = 10.0^(-13 + k)
tol_num = 13 # number of inner tolerances to test: inner_tol = 10.0^(-13 + k+j-1), j=1,...,tol_num+1-k 

B_svdt, svd_error, svd_ratios, singular_values, SigmaVt, square_sigma_ratio = svd_truncated(B, r)
save_singular_values_csv(r, singular_values, svd_ratios, size(B), cond_B)

outer_tol = 10.0^(-13 + k) 
resvec_inner = Vector{ALSResult}(undef, tol_num+1-k) # results for different inner tolerances
for j in range(1, tol_num+1-k)
    inner_tol = 10.0^(-13 + k+j-1) 
    res = als_2d_qr(B, r, cond_B, square_sigma_ratio, B_svdt, SigmaVt;
                outer_tol=outer_tol, inner_abstol=inner_tol, inner_reltol=0.0)
    resvec_inner[j] = res
end

outertolstr = @sprintf("outertol_%1.0e", outer_tol)
# Plot all curves in a single figure (fixed outertol and different innertol)
save_relaxation_panel_pdf(r, resvec_inner, svd_ratios, singular_values, outer_tol, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="QR_comparison_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")

# # create directory for "outlife" and save results
# # cg_method = "IterativeSolvers_cg"
# # matstr = "$(n)x$(m)"
# # condstr = @sprintf("cond%.2f", cond_B)
# # rankstr = "rank_$(r)"
# # outertolstr = @sprintf("outertol_%1.0e", outer_tol)
# # dir = "data/results_html/$(cg_method)/$(matstr)_$(condstr)/$(rankstr)/$(outertolstr)"
# # mkpath(dir)

# Plot each case separately (fixed outertol and different innertol)
save_rank_panel_pdf(r, resvec_inner, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="$(outertolstr)/QR_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol")
#----------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
# # Tests for a same rank r and different outer tolrances
# tol_num = 13
# r=3 # low-rank

# B_svdt, svd_error, svd_ratios, singular_values, SigmaVt, square_sigma_ratio = svd_truncated(B, r)
# save_singular_values_csv(r, singular_values, svd_ratios, size(B), cond_B)
# resvec = Vector{ALSResult}(undef, tol_num) # results for outer tolerance = inner tolerance

# for k in range(1, tol_num)
#     outer_tol = 10.0^(-13 + k) 
#     resvec_inner = Vector{ALSResult}(undef, tol_num+1-k)
#     for j in range(1, tol_num+1-k)
#         inner_tol = 10.0^(-13 + k+j-1) 
#         res = als_2d_qr(B, r, cond_B, square_sigma_ratio, B_svdt, SigmaVt;
#                     outer_tol=outer_tol, inner_abstol=inner_tol, inner_reltol=0.0)
#         resvec_inner[j] = res

#         if j == 1
#           resvec[k] = res
#          end
#     end
#     # Plot all curves in a single figure (fixed outertol and different innertol)
#     save_relaxation_panel_pdf(r, resvec_inner, svd_ratios, singular_values, outer_tol, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="QR_comparison_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")

#     # Plot each case separately (fixed outertol and different innertol)
#     outertolstr = @sprintf("outertol_%1.0e", outer_tol)
#     save_rank_panel_pdf(r, resvec_inner, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="$(outertolstr)/QR_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")
# end

# Plot exact results for outer_tol = inner_tol
# save_rank_panel_pdf(r, resvec, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="QR_exact")
#----------------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------------#
# # Tests with more ranks and different outer tolerances
# tol_num = 13
# #ranks = [3, 5, 7, 10]
# ranks = [3,5]

# for r in ranks
#     B_svdt, svd_error, svd_ratios, singular_values, SigmaVt, square_sigma_ratio = svd_truncated(B, r)
#     save_singular_values_csv(r, singular_values, svd_ratios, size(B), cond_B)
#     resvec = Vector{ALSResult}(undef, tol_num) # results for outer tolerance = inner tolerance
#     for k in range(1, tol_num)
#         outer_tol = 10.0^(-13 + k) 
#         resvec_inner = Vector{ALSResult}(undef, tol_num+1-k)
#         for j in range(1, tol_num+1-k)
#             inner_tol = 10.0^(-13 + k+j-1) 
#             res = als_2d_qr(B, r, cond_B, square_sigma_ratio, B_svdt, SigmaVt;
#                         outer_tol=outer_tol, inner_abstol=inner_tol, inner_reltol=0.0)
#             resvec_inner[j] = res
#             if j == 1
#                 resvec[k] = res
#             end
#         end
#         # Plot all curves in a single figure (fixed outertol and different innertol)
#         save_relaxation_panel_pdf(r, resvec_inner, svd_ratios, singular_values, outer_tol, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="QR_comparison_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")

#         # Plot each case separately (fixed outertol and different innertol)
#         outertolstr = @sprintf("outertol_%1.0e", outer_tol)
#         save_rank_panel_pdf(r, resvec_inner, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="$(outertolstr)/QR_rank_$(r)_fixed_outertol_$(outer_tol)_different_innertol_")
#     end
#     # Plot exact results (outer_tol = inner_tol)
#     save_rank_panel_pdf(r, resvec, svd_ratios, singular_values, size(B), cond_B; cg_method = "IterativeSolvers_cg", outfile="QR_exact")
# end
#----------------------------------------------------------------------------------------------------------------#