using JLD2
using LinearAlgebra

size_n = 1000
size_m = 1000
B = randn(size_n, size_m)
cond_B = cond(B)


isdir("../data/matrix") || mkpath("../data/matrix")

folder = "../data/matrix/$(size_n)x$(size_m)"
isdir(folder) || mkpath(folder)

cond_str = string(round(cond_B, digits=2))
filename = "$(folder)/$(size_n)x$(size_m)_cond$(cond_str).jld2"

try
    @save filename B cond_B
    println("Matrix saved to ", filename, ". Condition number: ", cond_B, " Size: ", size(B))
    println("File exists: ", isfile(filename))
    println("File size: ", filesize(filename), " bytes")
catch e
    println("Error saving file: ", e)
end