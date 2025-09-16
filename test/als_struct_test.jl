# using JLD2
# using Plots
# using Printf
# using Measures
# include("../src/als_struct.jl")
# include("../src/plots.jl")
# plotlyjs()
# data = load("data/matrix/10x10/10x10_cond9.24.jld2")
# B = data["B"]
# cond_B = data["cond_B"]
# println("condition number: ", cond_B)
# # rank = 1
# rank = 5
# B_svdt, svd_error = svd_truncated(B, rank)
# als_res= als_2d(B, rank, B_svdt)

# println("ALS Converged ?: ", als_res.converged_als)
# # println("outer error history: ", als_error_his)
    
# error1 = als_res.als_error[end]
# println("Final error from ALS: ", error1)
# error2 = svd_error
# println("Error of truncated SVD: ", error2)
# # error3 = norm(als_res.B_als - B_svdt, 2)
# # println("Error between ALS and truncated SVD: ", error3)

# x = range(1,als_res.outer_iters)
# y1 = als_res.als_error
# y2 = als_res.inner_iters
# y3 = als_res.time_cg

# # p1 = plot(range(1,als_res.outer_iters), als_res.als_error, xlabel="Outer Iterations", ylabel="Error at each outer step", title="Error History of ALS (rank=$rank)")
# # p2 = plot(range(1,als_res.outer_iters), als_res.inner_iters, xlabel="Outer Iterations", ylabel="Inner Iterations at each outer step", title="Inner Iteration History of ALS (rank=$rank)")
# # p3 = plot(range(1,als_res.outer_iters), als_res.time_cg, xlabel="Outer Iterations", ylabel="Time at each outer step", title="Time History of ALS (rank=$rank)")
# p1 = plot(x, y1, yscale=:log10, xlabel="Outer Iterations", ylabel="Error at each outer step", title="Error History of ALS (rank=$rank)")
# p2 = plot(x, y2, xlabel="Outer Iterations", ylabel="Inner Iterations at each outer step", title="Inner Iteration History of ALS (rank=$rank)")
# p3 = plot(x, y3, xlabel="Outer Iterations", ylabel="Time at each outer step", title="Time History of ALS (rank=$rank)")
# #p = plot(x, [y1 y2 y3], layout = (3,1), xlabel="Outer Iterations", legend = [:topright], label = ["Error at each outer step" "Inner Iterations at each outer step" "Time at each outer step(s)"], title="ALS Performance (rank=$rank) inner_tol=$(als_res.inner_tol) outer_tol=$(als_res.outer_tol)")
# display(p1)
# savefig(p1, "als_10x10_cond9.24_rank$rank _error.pdf")
# display(p2)
# savefig(p2, "als_10x10_cond9.24_rank$rank _inner_iters.pdf")
# display(p3)
# savefig(p3, "als_10x10_cond9.24_rank$rank _time.pdf")

# # savefig(p, "als_10x10_cond9.24_rank$rank.png")






# inner_tol = 10.0 .^range(-6, -12)
# outer_tol = 10.0 .^range(-6, -12)

# data = load("data/matrix/10x10/10x10_cond9.24.jld2")
# B = data["B"]
# cond_B = data["cond_B"]

# rank = [3, 5, 7, 10]
# B_svdt, svd_error = svd_truncated(B, rank)
# als_results = Dict{Int, ALSResult}()

# for r in rank
#     als_results[r] = als_2d(B, r, B_svdt)

#     println("ALS Converged ?: ", als_results[r].converged_als)
#     # println("outer error history: ", als_error_his)

#     x = range(1,als_results[r].outer_iters)
#     y1 = als_results[r].als_error
#     y2 = als_results[r].inner_iters
#     y3 = als_results[r].time_cg

#     # p1 = plot(range(1,als_res.outer_iters), als_res.als_error, xlabel="Outer Iterations", ylabel="Error at each outer step", title="Error History of ALS (rank=$rank)")
#     # p2 = plot(range(1,als_res.outer_iters), als_res.inner_iters, xlabel="Outer Iterations", ylabel="Inner Iterations at each outer step", title="Inner Iteration History of ALS (rank=$rank)")
#     # p3 = plot(range(1,als_res.outer_iters), als_res.time_cg, xlabel="Outer Iterations", ylabel="Time at each outer step", title="Time History of ALS (rank=$rank)")
#     p1 = plot(x, y1, yscale=:log10, xlabel="Outer Iterations", ylabel="Error at each outer step", title="Error History of ALS (rank=$r)")
#     p2 = plot(x, y2, xlabel="Outer Iterations", ylabel="Inner Iterations at each outer step", title="Inner Iteration History of ALS (rank=$r)")
#     p3 = plot(x, y3, xlabel="Outer Iterations", ylabel="Time at each outer step", title="Time History of ALS (rank=$r)")
#     #p = plot(x, [y1 y2 y3], layout = (3,1), xlabel="Outer Iterations", legend = [:topright], label = ["Error at each outer step" "Inner Iterations at each outer step" "Time at each outer step(s)"], title

# end 