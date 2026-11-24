# Short towers with long caps: can the floor be reached at 3 and 4 passes at all, staged or from scratch?
# Run: julia probe.jl KMAX nseeds cap
include(joinpath(@__DIR__, "passes.jl"))
function probe()
    KMAX = parse(Int, ARGS[1]); nseeds = parse(Int, ARGS[2]); cap = parse(Int, ARGS[3])
    work(k, T, s) = k * T * s
    println("== tower up to $KMAX passes on length $(KMAX+1), cap $cap steps per run or stage, lr 0.01 ==")
    println("regime seed  [per stage: steps(reached) birth_loss/floor final_loss/floor]  total_work=passes×positions×steps  all_reached")
    for seed in 1:nseeds
        p = init(MersenneTwister(seed)); tot = 0; ok = true
        print("staged  $seed ")
        for k in 1:KMAX
            p, s, r, ℓ0, a0, ℓ = train(p, k + 1, k; cap)
            fl = floor_exact(k + 1, k)
            @printf(" k=%d: %d(%s) %.2f %.3f |", k, s, r ? "y" : "n", ℓ0 / fl, ℓ / fl)
            tot += work(k, k + 1, s); ok &= r
        end
        println("  ", tot, "  ", ok); flush(stdout)
        p, s, r, ℓ0, a0, ℓ = train(init(MersenneTwister(seed)), KMAX + 1, KMAX; cap)
        @printf("scratch %d  k=%d: %d(%s) %.3f |  %d  %s\n", seed, KMAX, s, r ? "y" : "n", ℓ / floor_exact(KMAX + 1, KMAX), work(KMAX, KMAX + 1, s), r)
        flush(stdout)
    end
end
probe()
