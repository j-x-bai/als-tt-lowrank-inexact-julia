using Random
include("als_energy.jl")
include("plot_E_H.jl")
plotlyjs()
#gr()

# generate SPD H = I⊗A + A⊗I where A = SPD_Matrix is SPD
rng_A = MersenneTwister(2025)
rank = 25
n= 50
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
m = 50
I_m = Matrix{Float64}(I, m, m)
Kron_Product_1 = kron(I_m, A_spd) # Kronecker product: (m*n) x (m*n)
Kron_Product_2 =  kron(A_spd, I_m) # Kronecker product: (m*n) x (m*n)
Kron_Sum = Kron_Product_1 + Kron_Product_2
H = Symmetric(Kron_Sum)

#-------------------------------------------------------------------------------------------------------------#
# F_A = eigen(A_spd)
# V_A = F_A.vectors
# D_A = F_A.values

# # A^1/2, A^-1/2 and A^-1
# D_A_half = sqrt.(D_A)
# D_A_neg_half = 1 ./D_A_half 
# D_A_neg = 1 ./D_A
# A_half = V_A * Diagonal(D_A_half) * V_A'
# A_neg_half = V_A * Diagonal(D_A_neg_half) * V_A'
# A_neg = V_A * Diagonal(D_A_neg) * V_A'
#-------------------------------------------------------------------------------------------------------------#
F_H = eigen(H)
V_H = F_H.vectors
D_H = F_H.values

# H^{1/2} 
D_H_half = sqrt.(D_H)
H_half = V_H * Diagonal(D_H_half) * V_H'
println("largest eigenval of H^{1/2}: ", maximum(D_H_half)) 

# H^{-1/2} 
D_H_neg_half = 1.0 ./ D_H_half
H_neg_half = V_H * Diagonal(D_H_neg_half) * V_H'
println("largest eigenval of H^{-1/2}: ", maximum(D_H_neg_half)) 

# H^{-1} 
D_H_neg = 1.0 ./ D_H
H_neg = V_H * Diagonal(D_H_neg) * V_H'
println("largest eigenval of H^{-1}: ", maximum(D_H_neg)) 

#-------------------------------------------------------------------------------------------------------------#
# B : m x n
rng_B = MersenneTwister(444)
B = randn(rng_B, m, n)

#-------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------#

# NormalResult = als_2d_energy_spd_normal(A_spd, B, rank)
NormalResult = als_2d_energy_spd_normal_twice(A_spd, B, rank)
println("ALS Normal : rank = ", rank, " converged = ",NormalResult.converged_als," Outer iters = ", NormalResult.outer_iters, " total_inner_iters = ", NormalResult.total_inner_iters)
println("ALS Normal : last J_energy = ", NormalResult.J_energy[end]," last residual in frobenius norm = ", NormalResult.res_fro[end])

#-------------------------------------------------------------------------------------------------------------#

NewtonResult = als_2d_energy_spd_newton(A_spd, B, rank)
println("ALS Newton : rank = ", rank, " converged = ", NewtonResult.converged_als," Outer iters = ", NewtonResult.outer_iters, " total_inner_iters = ", NewtonResult.total_inner_iters)
println("ALS Newton : last J_energy = ", NewtonResult.J_energy[end]," last residual in frobenius norm = ", NewtonResult.res_fro[end])


#-------------------------------------------------------------------------------------------------------------#

html =  plot_energy_normal_and_newton_solution(
    NormalResult,   
    NewtonResult,           
    rank, 
    (m, n), 
    modified_energy;  
    cg_method = "IterativeSolvers_cg",
    name = name_A,
    outfile = "energy_vs_hnorm"
)
println("saved: ", html)