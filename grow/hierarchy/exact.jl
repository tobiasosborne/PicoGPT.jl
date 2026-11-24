include("world.jl")
using Printf
println("  ε    κ    δ   | coarse(T=8)  H(c'|c)   H(c'|v)   split    fine     defect    chain-rule residual")
for (ε, κ, δ) in [(0,0,0),(0,1,0),(1,1,0),(0,2,0),(1,2,0),(1,1,0.25),(1,1,0.5),(1,1,1.0),(1,1,2.0),(0,1,1.0)]
    e = entropies(world(; ε, κ, δ))
    @printf("%4.2f %4.2f %4.2f | %8.5f   %8.5f  %8.5f  %7.5f  %7.5f  %8.5f  %.1e\n", ε, κ, δ, e.coarse, e.c1, e.cfine, e.split, e.fine, e.defect, e.chain)
    @assert abs(e.chain) < 1e-12
    δ == 0 && @assert abs(e.defect) < 1e-12 && abs(e.coarse - e.c1) < 1e-12
end
println("chain rule fine = H(c'|v) + split verified to 1e-12; defect = 0 and coarse floor = H(c'|c) whenever δ = 0.")
