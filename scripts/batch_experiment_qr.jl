using JLD2, DataFrames, CSV, PrettyTables
using Base.Threads
# include("src/als_simple_qr.jl")
include(joinpath(@__DIR__, "../src/als_simple_qr.jl"))

num_threads = Threads.nthreads()

matrix_dirs = readdir(joinpath(@__DIR__, "../data/matrix"), join=true)
println("successful read matrix dirs: ", matrix_dirs)

for dir in matrix_dirs
    if isdir(dir)
        size_str = split(dir, "/")[end]
        # size_n, size_m = parse.(Int, split(size_str, "x"))
        local size_n, size_m = parse.(Int, split(size_str, "x"))

        # matrix_files = readdir(dir, join=true)
        matrix_files = filter(f -> endswith(f, ".jld2"), readdir(dir, join=true))
        println("matrix_files: ", matrix_files)

        isdir("data/results_qr") || mkpath("../data/results_qr")
        isdir("data/results_qr/results_qr_$(size_str)") || mkpath("data/results_qr/results_qr_$(size_str)")


        for file in matrix_files
            local B 
            local cond_B
            @load file B cond_B
            local cond_str = string(round(cond_B, digits=2))

            n, m = size(B)
            @assert size_n == n && size_m == m

            isdir("data/results_qr/results_qr_$(size_str)/qr_cond_$(cond_str)") || mkpath("data/results_qr/results_qr_$(size_str)/qr_cond_$(cond_str)")

            if isfile("data/results_qr/results_qr_$(size_str)/qr_cond_$(cond_str)/qr_cond_$(cond_str).csv")
                results = CSV.read("data/results_qr/results_qr_$(size_str)/qr_cond_$(cond_str)/cond_$(cond_str).csv", DataFrame)
            else
                results = DataFrame(
                    threads = Int[],
                    n = Int[],
                    m = Int[],
                    cond_num = Float64[],
                    rank = Int[],
                    outer_tol = Float64[],
                    outer_maxiters = Int[],
                    inner_tol = Float64[],
                    inner_maxiters = Int[],
                    als_error = Float64[],
                    svdt_error = Float64[],
                    als_svdt_error = Float64[],
                    converged_als = Bool[],
                    als_time = Float64[],
                    svdt_time = Float64[]
                )
            end

            for rank in [ceil(Int, 0.25*m), 
                ceil(Int, 0.5*m), ceil(Int, 0.75*m), m
                ]

                svdt_time = @elapsed B_svdt, singular_values = svd_truncated(B, rank)

                for outer_tol in [
                    1e-6, 
                    #1e-8
                    ]
                    for outer_maxiters in [25, 
                        50, 75, 100, 
                        #150, 200, 250, 300
                        ]
                        for inner_tol in [
                            1e-6, 
                            #1e-8
                            ]
                            for inner_maxiters in [25, 
                                50, 75, 100, 
                                # 150, 200, 250, 300
                                ]

                                als_time = @elapsed B_als, als_error_his, converged_als = als_2d_qr(B, rank; inner_maxiters=inner_maxiters, inner_tol=inner_tol, outer_maxiters=outer_maxiters, outer_tol=outer_tol)
                                # svdt_time = @elapsed B_svdt, singular_values = svd_truncated(B, rank)

                                als_error = als_error_his[end]
                                svdt_error = norm(B_svdt - B, 2)
                                als_svdt_error = norm(B_als - B_svdt, 2)

                                # push!(results, (
                                #     num_threads,      
                                #     n,                
                                #     m,                
                                #     cond_B,           
                                #     rank,             
                                #     outer_tol,        
                                #     outer_maxiters,   
                                #     inner_tol,        
                                #     inner_maxiters,   
                                #     als_error,        
                                #     svdt_error,       
                                #     als_svdt_error,  
                                #     converged_als,
                                #     als_time,         
                                #     svdt_time         
                                # ))

                                push!(results, (
                                    threads = num_threads,      
                                    n = n,                
                                    m = m,                
                                    cond_num = cond_B,           
                                    rank = rank,             
                                    outer_tol = outer_tol,        
                                    outer_maxiters = outer_maxiters,   
                                    inner_tol = inner_tol,        
                                    inner_maxiters = inner_maxiters,   
                                    als_error = als_error,        
                                    svdt_error = svdt_error,       
                                    als_svdt_error = als_svdt_error,  
                                    converged_als = converged_als,
                                    als_time = als_time,         
                                    svdt_time = svdt_time         
                                ))

                                
                                CSV.write("data/results_qr/results_qr_$(size_str)/qr_cond_$(cond_str)/qr_cond_$(cond_str).csv", results)
                                open("data/results_qr/results_qr_$(size_str)/qr_cond_$(cond_str)/qr_cond_$(cond_str).tex", "w") do io

                                    println(io, "\\documentclass{article}")
                                    println(io, "\\usepackage[a4paper,landscape,margin=0.5cm]{geometry}")
                                    println(io, "\\usepackage{booktabs}")
                                    println(io, "\\usepackage{longtable}")
                                    println(io, "\\begin{document}")

                                    println(io, "\\section*{QR Size: $(size_str), Condition Number: $(cond_B)}")
                                    println(io, "\\footnotesize")
                                    pretty_table(io, results, backend=Val(:latex), table_type=:longtable)
                                    println(io, "\\newpage")

                                    println(io, "\\end{document}")

                                end

                            end

                        end
                    end
                end
            end
        end
    end 
    
end