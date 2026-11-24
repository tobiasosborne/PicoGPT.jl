# Control for grow/doors: two-pass model from random weights on the fine world,
# gate starting at 0 (as the children) instead of 1. Same init and data stream as scratch-2pass.
# Needs a checkout of origin/grow/doors (its path is the first argument), about 1 min per seed:
#   OPENBLAS_NUM_THREADS=1 julia --project=<checkout> -t1 doors_gate0.jl <checkout> 1:9 4000 0.0
# Output (doors_gate0.txt): seed, gate at start, steps to 40/50/90/99 %, steps run, final and
# best test accuracy, final gate. The seed-1 row with gate 1.0 reproduces the report's scratch run.
include(joinpath(ARGS[1], "grow/doors/doors.jl"))
seeds = (r = parse.(Int, split(ARGS[2], ":")); r[1]:r[end]); cap = parse(Int, ARGS[3])
g0s = parse.(Float64, split(ARGS[4], ","))
c = config(24, 1, 48)
for seed in seeds, g0 in g0s
    rng = Xoshiro(seed); testf = fine(Xoshiro(1000 + seed), 1000)
    θ, cv = train!(newθ(c, rng; g = g0), fine, c, 2; rng, test = testf, steps = cap, stop = 0.995)
    println(join((seed, g0, reach(cv, 0.40), reach(cv, 0.50), reach(cv, 0.90), reach(cv, 0.99), cv[end][1], cv[end][3], maximum(x[3] for x in cv), round(θ[end], digits=3)), "\t")); flush(stdout)
end
