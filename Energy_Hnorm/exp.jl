using Random
include("als_Hnorm.jl")
include("als_energy.jl")
include("plot_E_H.jl")
plotlyjs()

# generate SPD H = I⊗A + A⊗I where A = SPD_Matrix is SPD
rng_A = MersenneTwister(2025)
rank = 40


n= 80
modified_energy = false
# randn A 
A = randn(rng_A, n, n) 
# A_spd = A' * A
A_spd = Symmetric(A' * A)
SPD_Matrix = Symmetric(A_spd)
name_A = "A=randn"

# # A = Id
# A_spd = Matrix{Float64}(I, n, n)
# name_A = "A=Id"

# # A = 0.5*Id
# A_spd = 0.5.*Matrix{Float64}(I, n, n)
# name_A = "A=0.5Id"

# # A = diag([base^i])
# base = 2
# diag_values = [base^i for i in 1:n]
# A_spd = Diagonal(diag_values)
# name_A = "A=diag"

#-------------------------------------------------------------------------------------------------------------#
# H
m = 80
I_m = Matrix{Float64}(I, m, m)
Kron_Product_1 = kron(I_m, A_spd) # Kronecker product: (m*n) x (m*n)
Kron_Product_2 =  kron(A_spd, I_m) # Kronecker product: (m*n) x (m*n)
Kron_Sum = Kron_Product_1 + Kron_Product_2
H = Symmetric(Kron_Sum)

#-------------------------------------------------------------------------------------------------------------#
F_H = eigen(H)
V_H = F_H.vectors
D_H = F_H.values

# H^{1/2} 
D_H_half = sqrt.(D_H)
H_half = V_H * Diagonal(D_H_half) * V_H'
println("The largest eigenval of H^{1/2}: ", maximum(D_H_half)) 

# H^{-1/2} 
D_H_neg_half = 1.0 ./ D_H_half
H_neg_half = V_H * Diagonal(D_H_neg_half) * V_H'
println("The largest eigenval of H^{-1/2}: ", maximum(D_H_neg_half)) 

# H^{-1} 
D_H_neg = 1.0 ./ D_H
H_neg = V_H * Diagonal(D_H_neg) * V_H'
println("The largest eigenval of H^{-1}: ", maximum(D_H_neg)) 
#-------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------#
# B : m x n
rng_B = MersenneTwister(444)
B = randn(rng_B, m, n)
#-------------------------------------------------------------------------------------------------------------#
vec_B = reshape(B, m*n, 1)
vec_hat_B = H_neg_half * vec_B
hat_B = reshape(vec_hat_B, m, n) 
cond_hat_B = cond(hat_B)
hat_B_fnorm = norm(hat_B, 2)

# H norm solition
hat_B_svd_truncated, hat_B_svd_error, hat_B_ratios, hat_B_singular_values, hat_B_SigmaVt, hat_B_square_sigma_ratio = svd_truncated(hat_B, rank)
hat_ALSResult = als_2d_qr(hat_B, rank, cond_hat_B, hat_B_square_sigma_ratio, hat_B_svd_truncated,hat_B_SigmaVt)
hat_X_sol = hat_ALSResult.X1_sol * (hat_ALSResult.X2_sol')
println("The last ALS error in H norm = ", hat_ALSResult.als_error[end])

vec_hat_X_sol = reshape(hat_X_sol, m*n, 1)
vec_X_Hnorm_sol = H_neg_half * vec_hat_X_sol
X_Hnorm_sol = reshape(vec_X_Hnorm_sol, m, n) 

#-------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------#
vec_view_hat_B_svd_truncated = reshape(hat_B_svd_truncated, m*n, 1)
vec_B_Hnrom_truncated_sol = H_neg_half * vec_view_hat_B_svd_truncated
B_Hnrom_truncated_sol = reshape(vec_B_Hnrom_truncated_sol, m, n) 
#-------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------#
# Energy solution
if modified_energy
    EResult = als_2d_modified_energy_spd(A_spd, B, hat_B_fnorm, rank)
else
    EResult = als_2d_energy_spd(A_spd, B, rank)
end
# EResult =  als_2d_modified_energy_spd(A_spd, A_neg_half, B, rank)
println("Energy solution: rank = ", rank, " converged = ",EResult.converged_als," Outer iters = ", EResult.outer_iters, " total_inner_iters = ", EResult.total_inner_iters)
println("Energy solution: last J_energy = ", EResult.J_energy[end]," last residual in frobenius norm = ", EResult.res_fro[end])

X_energy_sol = EResult.X1_sol * (EResult.X2_sol')

error_energy_and_X_Hnorm_sol = norm(X_energy_sol - X_Hnorm_sol, 2)
error_energy_and_B_Hnrom_truncated_sol = norm(X_energy_sol - B_Hnrom_truncated_sol, 2)
println("The error between energy and X_Hnorm_sol = ", error_energy_and_X_Hnorm_sol)
println("The error between energy and B_Hnrom_truncated_sol = ", error_energy_and_B_Hnrom_truncated_sol)
#-------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------#
vec_X_energy_sol = reshape(X_energy_sol, m*n, 1)
vec_hat_X_energy_sol = H_half * vec_X_energy_sol
hat_X_energy_sol = reshape(vec_hat_X_energy_sol, m, n) 

error_hat_X_energy_sol_and_hat_X_sol = norm(hat_X_energy_sol - hat_X_sol, 2)
error_hat_X_energy_sol_and_hat_B_svd_truncated = norm(hat_X_energy_sol - hat_B_svd_truncated, 2)
#-------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------#
vec_B = reshape(B, m*n, 1)
vec_invHB = H_neg * vec_B 
invHB = reshape(vec_invHB, m, n) 
invHB_svd_truncated, invHB_svd_error, invHB_ratios, invHB_singular_values, invHB_SigmaVt, invHB_square_sigma_ratio = svd_truncated(invHB, rank)
#-------------------------------------------------------------------------------------------------------------#
error_energy_and_invHB_svd_truncated = norm(X_energy_sol - invHB_svd_truncated, 2)
error_X_Hnorm_sol_and_invHB_svd_truncated = norm(X_Hnorm_sol - invHB_svd_truncated, 2)
error_B_Hnrom_truncated_sol_and_invHB_svd_truncated = norm(invHB_svd_truncated - B_Hnrom_truncated_sol, 2)

println("The error between energy and invHB_svd_truncated = ", error_energy_and_invHB_svd_truncated)
println("The error between X_Hnorm_sol and invHB_svd_truncated = ", error_X_Hnorm_sol_and_invHB_svd_truncated)
println("The error between invHB_svd_truncated and B_Hnrom_truncated_sol = ", error_B_Hnrom_truncated_sol_and_invHB_svd_truncated)
#-------------------------------------------------------------------------------------------------------------#

html =  plot_energy_and_Hnorm_solution(
    EResult,              
    hat_ALSResult,         
    rank,
    hat_B_fnorm,
    hat_B_singular_values,
    (m, n), 
    modified_energy,  
    B, H_neg_half, H_neg;  
    cg_method = "IterativeSolvers_cg",
    name = name_A,
    outfile = "energy_vs_hnorm"
)
println("saved: ", html)



