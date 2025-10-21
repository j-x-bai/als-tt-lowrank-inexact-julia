using LinearAlgebra
using ForwardDiff
using LaTeXStrings
include("plots_NM.jl")


function newton_global_2d(f::Function;
                       initial_points=nothing,
                       maxiters::Int=500,
                       tol::Float64=1e-12)

    if initial_points === nothing
        initial_points = zeros(2)
    end
    z = copy(initial_points)

    vec_x = zeros(maxiters)
    vec_y = zeros(maxiters)
    g_1 = zeros(maxiters)
    g_2 = zeros(maxiters)
    delta_x = zeros(maxiters)
    delta_y = zeros(maxiters)
    H11 = zeros(maxiters)
    H12 = zeros(maxiters)
    H21 = zeros(maxiters)
    H22 = zeros(maxiters)
    f_vals = zeros(maxiters)
    f_change  = zeros(maxiters)

    f_change[1] = f(initial_points)
    f_vals[1] = f(initial_points)
    delta_x[1] = NaN
    delta_y[1] = NaN

    converged = false
    iters = 0
    z_new = zeros(2)
    vec_x[1] = z[1]
    vec_y[1] = z[2]

    for k in 2:maxiters
        iters += 1
        g  = ForwardDiff.gradient(f, z)
        H  = ForwardDiff.hessian(f, z)

        g_1[k-1]=g[1]; g_2[k-1]=g[2]
        H11[k-1] = H[1,1]; H12[k-1] = H[1,2]; H21[k-1] = H[2,1]; H22[k-1] = H[2,2]

        # p = try
        #     - (cholesky(H, check=true) \ g)
        # catch
        #     - ((H + 1e-6I) \ g)
        # end
        # p = -inv(H)*g
        p = -(H \ g)

        z_new = z + p
        vec_x[k] = z_new[1]
        vec_y[k] = z_new[2]
        # vec_z[:, k] = z_new
        f_vals[k] = f(z_new)
        f_change[k] = abs(f_vals[k] - f_vals[k-1])
        delta_x[k] = p[1]  # negative
        delta_y[k] = p[2]
        
        grad = ForwardDiff.gradient(f, z_new)
        grad_cond = norm(grad, Inf) ≤ tol
        step_cond = norm(z_new - z, Inf) ≤ tol * (1 + norm(z_new, Inf))
        frel_cond = f_change[k] ≤ tol * (f_change[k-1] + 1)

        
        z .= z_new
        if grad_cond && step_cond && frel_cond
            converged = true
            break
        end
    end

    resize!(vec_x, iters+1)
    resize!(vec_y, iters+1)
    resize!(g_1, iters+1)
    resize!(g_2, iters+1)
    resize!(delta_x, iters+1)
    resize!(delta_y, iters+1)
    resize!(f_vals, iters+1)
    resize!(H11, iters+1)
    resize!(H12, iters+1)
    resize!(H21, iters+1)
    resize!(H22, iters+1)
    resize!(f_change, iters+1)

    return NewtonResult(vec_x, vec_y, f_vals, g_1, g_2, delta_x, delta_y, H11, H12, H21, H22, f_change, tol, iters, converged)
end


function newton_als_2d(f::Function;
                       initial_points=nothing,
                       maxiters::Int=500,
                       tol::Float64=1e-12)

    if initial_points === nothing
        initial_points = zeros(2)
    end
    z = copy(initial_points)

    vec_x = zeros(maxiters)
    vec_y = zeros(maxiters)
    g_1 = zeros(maxiters)
    g_2 = zeros(maxiters)
    delta_x = zeros(maxiters)
    delta_y = zeros(maxiters)
    H11 = zeros(maxiters)
    H12 = zeros(maxiters)
    H21 = zeros(maxiters)
    H22 = zeros(maxiters)
    f_vals = zeros(maxiters)
    f_change  = zeros(maxiters)

    f_change[1] = f(initial_points)
    f_vals[1] = f(initial_points)
    delta_x[1] = NaN
    delta_y[1] = NaN

    converged = false
    iters = 0

    z_half = zeros(2)
    z_new=zeros(2)
    vec_x[1] = z[1]
    vec_y[1] = z[2]

    for k in 2:maxiters
        iters += 1

        g = ForwardDiff.gradient(f, z)
        g_1[k-1] = g[1]; g_2[k-1] = g[2]
        H = ForwardDiff.hessian(f, z)
        H11[k-1] = H[1,1]; H12[k-1] = H[1,2]; H21[k-1] = H[2,1]; H22[k-1] = H[2,2]
        g_x = g[1]
        H_xx = H[1,1]
        p_x = -(H_xx \ g_x)
        x_new = z[1] + p_x
        delta_x[k] = p_x  # negative

        
        z_half[1] = x_new; z_half[2] = z[2]
        g_y = ForwardDiff.gradient(f, z_half)[2]
        H_yy = ForwardDiff.hessian(f, z_half)[2,2]
        p_y = -(H_yy \ g_y)
        y_new = z[2] + p_y
        delta_y[k] = p_y

        z_new[1] = x_new; z_new[2] = y_new
        vec_x[k] = x_new
        vec_y[k] = y_new
        f_vals[k] = f(z_new)
        f_change[k] = abs(f_vals[k] - f_vals[k-1])

        grad = ForwardDiff.gradient(f, z_new)
        grad_cond = norm(grad, Inf) ≤ tol
        step_cond = norm(z_new - z, Inf) ≤ tol * (1 + norm(z_new, Inf))
        frel_cond = f_change[k] ≤ tol * (abs(f_vals[k-1]) + 1)
        
        if grad_cond && step_cond && frel_cond
            converged = true
            z .= z_new
            break
        end
        z .= z_new
    end

    resize!(vec_x, iters+1)
    resize!(vec_y, iters+1)
    resize!(g_1, iters+1)
    resize!(g_2, iters+1)
    resize!(delta_x, iters+1)
    resize!(delta_y, iters+1)
    resize!(f_vals, iters+1)
    resize!(H11, iters+1)
    resize!(H12, iters+1)
    resize!(H21, iters+1)
    resize!(H22, iters+1)
    resize!(f_change, iters+1)

    return NewtonResult(vec_x, vec_y, f_vals, g_1, g_2, delta_x, delta_y, H11, H12, H21, H22, f_change, tol, iters, converged)
end





function newton_als_2d_full(f::Function;
                       initial_points=nothing,
                       maxiters::Int=1000,
                       tol::Float64=1e-12)

    if initial_points === nothing
        initial_points = zeros(2)
    end
    z = copy(initial_points)

    vec_x = zeros(maxiters)
    vec_y = zeros(maxiters)
    g_1 = zeros(maxiters)
    g_2 = zeros(maxiters)
    delta_x = zeros(maxiters)
    delta_y = zeros(maxiters)
    H11 = zeros(maxiters)
    H12 = zeros(maxiters)
    H21 = zeros(maxiters)
    H22 = zeros(maxiters)
    f_vals = zeros(maxiters)
    f_change  = zeros(maxiters)

    f_change[1] = f(initial_points)
    f_vals[1] = f(initial_points)
    delta_x[1] = NaN
    delta_y[1] = NaN
    converged = false
    iters = 0
    z_half = zeros(2)
    z_new=zeros(2)
    vec_x[1] = z[1]
    vec_y[1] = z[2]






    vec_x_full = zeros(maxiters*2)
    vec_y_full = zeros(maxiters*2)
    g_1_full = zeros(maxiters*2)
    g_2_full = zeros(maxiters*2)
    delta_x_full = zeros(maxiters*2)
    delta_y_full = zeros(maxiters*2)
    H11_full = zeros(maxiters*2)
    H12_full = zeros(maxiters*2)
    H21_full = zeros(maxiters*2)
    H22_full = zeros(maxiters*2)
    f_vals_full = zeros(maxiters*2)
    f_change_full  = zeros(maxiters*2)

    f_change_full[1] = f(initial_points)
    f_vals_full[1] = f(initial_points)
    delta_x_full[1] = NaN
    delta_y_full[1] = NaN
    iters_full = 0
    vec_x_full[1] = z[1]
    vec_y_full[1] = z[2]






    for k in 2:maxiters
        iters += 1
        iters_full +=2
        k_full = (k-1) *2

        g = ForwardDiff.gradient(f, z)
        g_1[k-1] = g[1]; g_2[k-1] = g[2]
        g_1_full[k_full-1] = g[1]; g_2_full[k_full-1] = g[2]
        H = ForwardDiff.hessian(f, z)
        H11[k-1] = H[1,1]; H12[k-1] = H[1,2]; H21[k-1] = H[2,1]; H22[k-1] = H[2,2]
        H11_full[k_full-1] = H[1,1]; H12_full[k_full-1] = H[1,2]; H21_full[k_full-1] = H[2,1]; H22_full[k_full-1] = H[2,2]
        g_x = g[1]
        H_xx = H[1,1]
        p_x = -(H_xx \ g_x)
        x_new = z[1] + p_x
        delta_x[k] = p_x  # negative
        delta_x_full[k_full] = p_x
        delta_x_full[k_full+1] = 0

        
        z_half[1] = x_new; z_half[2] = z[2]
        vec_x_full[k_full] = x_new
        vec_y_full[k_full] = z[2]
        g_half = ForwardDiff.gradient(f, z_half)
        g_1_full[k_full] = g_half[1]; g_2_full[k_full] = g_half[2]
        H_half = ForwardDiff.hessian(f, z_half)
        H11_full[k_full] = H_half[1,1]; H12_full[k_full] = H_half[1,2]; H21_full[k_full] = H_half[2,1]; H22_full[k_full] = H_half[2,2]
        f_vals_full[k_full] = f(z_half)
        f_change_full[k_full] = abs(f_vals_full[k_full] - f_vals_full[k_full-1])


        g_y = ForwardDiff.gradient(f, z_half)[2]
        H_yy = ForwardDiff.hessian(f, z_half)[2,2]
        p_y = -(H_yy \ g_y)
        y_new = z[2] + p_y
        delta_y[k] = p_y
        delta_y_full[k_full] = 0
        delta_y_full[k_full+1] = p_y

        z_new[1] = x_new; z_new[2] = y_new
        vec_x[k] = x_new
        vec_y[k] = y_new
        vec_x_full[k_full+1] = x_new
        vec_y_full[k_full+1] = y_new

        f_vals[k] = f(z_new)
        f_vals_full[k_full+1] = f_vals[k]
        f_change[k] = abs(f_vals[k] - f_vals[k-1])
        f_change_full[k_full+1] = abs(f_vals_full[k_full+1] - f_vals_full[k_full])

        grad = ForwardDiff.gradient(f, z_new)
        grad_cond = norm(grad, Inf) ≤ tol
        step_cond = norm(z_new - z, Inf) ≤ tol * (1 + norm(z_new, Inf))
        frel_cond = f_change[k] ≤ tol * (abs(f_vals[k-1]) + 1)
        
        if grad_cond && step_cond && frel_cond
            converged = true
            z .= z_new
            break
        end
        z .= z_new
    end

    resize!(vec_x, iters+1)
    resize!(vec_y, iters+1)
    resize!(g_1, iters+1)
    resize!(g_2, iters+1)
    resize!(delta_x, iters+1)
    resize!(delta_y, iters+1)
    resize!(f_vals, iters+1)
    resize!(H11, iters+1)
    resize!(H12, iters+1)
    resize!(H21, iters+1)
    resize!(H22, iters+1)
    resize!(f_change, iters+1)

    resize!(vec_x_full, iters_full+1)
    resize!(vec_y_full, iters_full+1)
    resize!(g_1_full, iters_full+1)
    resize!(g_2_full, iters_full+1)
    resize!(delta_x_full, iters_full+1)
    resize!(delta_y_full, iters_full+1)
    resize!(f_vals_full, iters_full+1)
    resize!(H11_full, iters_full+1)
    resize!(H12_full, iters_full+1)
    resize!(H21_full, iters_full+1)
    resize!(H22_full, iters_full+1)
    resize!(f_change_full, iters_full+1)

    return NewtonResult(vec_x, vec_y, f_vals, g_1, g_2, delta_x, delta_y, H11, H12, H21, H22, f_change, tol, iters, converged), NewtonResult(vec_x_full, vec_y_full, f_vals_full, g_1_full, g_2_full, delta_x_full, delta_y_full, H11_full, H12_full, H21_full, H22_full, f_change_full, tol, iters_full, converged)
end


# vec_epi = 1:10:100 # ALS_NM will be converged at one step!!
# vec_epi_name="1to100"
# vec_epi = 0.1:0.1:2 # Global_NM and ALS_NM have same convergence
# vec_epi_name="01to2"
vec_epi = 1:20:100 # Global_NM and ALS_NM have same convergence
vec_epi_name="1to100by20"
Global_NMResults = Dict{Float64, NewtonResult}()
ALS_NMResults = Dict{Float64, NewtonResult}()
ALS_full_NMResults = Dict{Float64, NewtonResult}()
Error_f_Global_ALS_NM = Dict{Float64, Float64}()
Error_y_Global_ALS_NM = Dict{Float64, Vector{Float64}}()
f_vec = Dict{Float64, Function}()
f_name_vec = Dict{Float64, LaTeXString}()

# ALSSYM_NMResults = Dict{Float64, NewtonResult}()
for epi in vec_epi
    z_0 = zeros(2)
    epi_val_str = @sprintf("%.4f", epi)
    #f(z) = (z[1]-10)^6 + (z[2]-10)^6 + epi*z[1]*z[2] # constant hessian
    f(z) = (z[1]-10)^4 + (z[2]-10)^4 + epi*(z[1])^2*(z[2])^2 
    f_name = L"f(x,y) = (x - 10)^4 + (y - 10)^4 + %$(epi_val_str) x^2 y^2"
    f_vec[epi] = f
    f_name_vec[epi] =f_name

    Global_NMResults[epi] = newton_global_2d(f; initial_points = z_0)
    # println("Global_NM solution= ", Global_NMResults[epi].vec_z)
    println("Global_NM total iters= ", Global_NMResults[epi].iters)
    # println("Global_NM H11= ", Global_NMResults[epi].H11)
    # println("Global_NM H22= ", Global_NMResults[epi].H22)
    # println("Global_NM H12= ", Global_NMResults[epi].H12)
    # println("Global_NM H21= ", Global_NMResults[epi].H21)
    # println("Global_NM g_1= ", Global_NMResults[epi].g_1)
    # println("Global_NM g_2= ", Global_NMResults[epi].g_2)
    # ALS_NMResults[epi] = newton_als_2d(f; initial_points = z_0)
    # println("ALS_NM solution= ", ALS_NMResults[epi].vec_z)
    # println("ALS_NM total iters= ", ALS_NMResults[epi].iters)
    # println("ALS_NM H11= ", ALS_NMResults[epi].H11)
    # println("ALS_NM H22= ", ALS_NMResults[epi].H22)
    # println("ALS_NM H12= ", ALS_NMResults[epi].H12)
    # println("ALS_NM H21= ", ALS_NMResults[epi].H21)
    # println("ALS_NM g_1= ", ALS_NMResults[epi].g_1)
    # println("ALS_NM g_2= ", ALS_NMResults[epi].g_2)
    ALS_NMResults[epi], ALS_full_NMResults[epi] = newton_als_2d_full(f; initial_points = z_0)

    println("Global_NM last f= ", Global_NMResults[epi].f_vals[end])
    println("ALS_NM last f= ", ALS_NMResults[epi].f_vals[end])
    Error_f_Global_ALS_NM[epi] = abs(ALS_NMResults[epi].f_vals[end] - Global_NMResults[epi].f_vals[end])
    println("last f error between Global_NM and ALS_NM= ", Error_f_Global_ALS_NM[epi])

    # Error_y_Global_ALS_NM[epi] = Global_NMResults[epi].vec_z[2,:] - ALS_NMResults[epi].vec_z[2,:] # not a same size.
    # println("y error between Global_NM and ALS_NM=", Error_y_Global_ALS_NM[epi])
    # println("theorical delta y=", ALS_NMResults[epi].delta_y)
end

# NM_panel_pdf(Global_NMResults, ALS_NMResults, vec_epi,
#             f_vec,f_name_vec; f_name="f4_sym", vec_epi_name=vec_epi_name)

# NM_panel_pdf_full(Global_NMResults, ALS_NMResults, ALS_full_NMResults, vec_epi, 
#             f_vec,f_name_vec; f_name="f4_sym", vec_epi_name=vec_epi_name)
# plot_2d_solution(Global_NMResults, ALS_NMResults, vec_epi,f_vec,f_name_vec; f_name="f4_sym", vec_epi_name=vec_epi_name)

# NM_panel_pdf_log(Global_NMResults, ALS_NMResults, vec_epi,
#             f_vec,f_name_vec; f_name="f4_sym", vec_epi_name=vec_epi_name)

# NM_panel_pdf_full_log(Global_NMResults, ALS_NMResults, ALS_full_NMResults, vec_epi, 
#             f_vec,f_name_vec; f_name="f4_sym", vec_epi_name=vec_epi_name)


plot_2d_solution(Global_NMResults, ALS_NMResults, vec_epi,f_vec,f_name_vec; f_name="f4_sym", vec_epi_name=vec_epi_name)
