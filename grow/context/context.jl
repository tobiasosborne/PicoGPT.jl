# Longer context: what a model trained at T=8 inherits at T=16 and T=32.
#   OPENBLAS_NUM_THREADS=1 julia -t2 --project grow/context/context.jl
# World: token = (a, b), a ∈ 1:A, b ∈ 1:Bn, two independent processes
#   a_n = a_{n-3}  w.p. 1-ε, else uniform on 1:A      (short dependency, visible at T=8)
#   b_n = b_{n-12} w.p. 1-δ, else uniform on 1:Bn     (long dependency, invisible at T=8)
# Each is a bundle of independent lag-chains with uniform stationary law, so the exact
# floor for predicting the token after position t of a window (t tokens seen) is
#   floor(t) = (t ≥ 3 ? h(ε,A) : ln A) + (t ≥ 12 ? h(δ,Bn) : ln Bn).
using PicoGPT, Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)

const A, Bn, LA, LB, ε, δ = 4, 4, 3, 12, 0.1, 0.1
const V = A * Bn
h(e, K) = (p = 1 - e + e / K; q = e / K; -(p * log(p) + (K - 1) * q * log(q)))
floorat(t) = (t ≥ LA ? h(ε, A) : log(A)) + (t ≥ LB ? h(δ, Bn) : log(Bn))

function stream(n, rng)
    a = rand(rng, 1:A, n); b = rand(rng, 1:Bn, n)
    for i in 1:n
        i > LA && rand(rng) > ε && (a[i] = a[i-LA])
        i > LB && rand(rng) > δ && (b[i] = b[i-LB])
    end
    (a .- 1) .* Bn .+ b
end

# per-position mean surprise, T-vector, of model θ run on windows x (T×B) with targets y
function perpos(θ, x, y, c)
    T, B = size(x)
    Z, _ = gpt(θ, x, c)
    s = zeros(T)
    for b in 1:B, t in 1:T
        z = @view Z[:, (b-1)*T+t]
        s[t] += logsumexp(z) - z[y[t, b]]
    end
    s ./ B
end

# sliding window: the prediction after position t uses only the last W tokens
function sliding(θ, x, y, c, W)
    T, B = size(x)
    s = zeros(T)
    for t in 1:T
        lo = max(1, t - W + 1)
        s[t] = perpos(θ, x[lo:t, :], y[lo:t, :], c)[end]
    end
    s
end

cfg(T) = Config(; V, D = 32, H = 2, L = 1, T, F = 64)

function train!(θ, opt, data, T, steps, rng; B = 32, η = 2e-3, every = 0, evalf = nothing, log = nothing, work0 = 0)
    c = cfg(T)
    for s in 1:steps
        x, y = windows(data, T, rand(rng, 1:length(data)-T, B))
        step!(θ, opt, x, y, c; η = η * min(1, s / 50))
        if every > 0 && s % every == 0
            push!(log, (work0 + s * B * T, evalf(θ)))
        end
    end
    θ
end

rng = MersenneTwister(1)
data = stream(200_000, rng)
test = stream(100_000, MersenneTwister(2))
Tmax = 32
xe, ye = windows(test, Tmax, 1:124:(length(test)-Tmax))     # 807 test windows
fl = floorat.(1:Tmax)
excess(θ, T = Tmax) = perpos(θ, xe[1:T, :], ye[1:T, :], cfg(T)) .- fl[1:T]
println("nparams = ", nparams(cfg(8)), " (same for every T: ", nparams(cfg(32)) == nparams(cfg(8)), ")")
@printf("floors: t<3 %.4f, 3≤t<12 %.4f, t≥12 %.4f nats\n", fl[1], fl[3], fl[12])

STEPS = parse(Int, get(ENV, "STEPS", "3000"))
models = Dict{Int,Vector{Float64}}()
logs = Dict{Int,Any}()
for T in (8, 16, 32)
    θ = init(cfg(T); rng = MersenneTwister(10))               # same start for all three
    logs[T] = Tuple{Int,Float64}[]
    t0 = time()
    train!(θ, AdamW(θ), data, T, STEPS, MersenneTwister(20 + T);
           every = 100, evalf = θ -> sum(excess(θ)) / Tmax, log = logs[T])
    models[T] = θ
    @printf("trained T=%d  %d steps  %.0f s  tokens %d\n", T, STEPS, time() - t0, STEPS * 32 * T)
end

# 1. the lift T → 2T is the identity on θ; predictions at positions 1..T do not change
θ8 = models[8]
for T2 in (16, 32)
    Zs, _ = gpt(θ8, xe[1:8, 1:200], cfg(8))
    Zl, _ = gpt(θ8, xe[1:T2, 1:200], cfg(T2))
    Zl = reshape(Zl, V, T2, 200)[:, 1:8, :]
    @printf("lift 8→%d: max |Δlogit| at positions 1..8 = %.3g\n", T2, maximum(abs, Zl .- reshape(Zs, V, 8, 200)))
end

# 2. per-position excess over the exact floor (nats), on the same test windows
g  = excess(models[8])                                        # T=8 model run at T=32, untrained
sl = sliding(models[8], xe, ye, cfg(8), 8) .- fl
s16 = excess(models[16], 16); s16x = excess(models[16])
s32 = excess(models[32])
println("\nexcess over floor per position (nats); floor drops by ", round(log(Bn) - h(δ, Bn), digits = 4), " at t=12")
println("  t   floor  T8@32  T8slide  scratch16(@32)  scratch32")
for t in 1:Tmax
    @printf("%3d  %.4f  %6.3f  %6.3f   %6.3f          %6.3f\n", t, fl[t], g[t], sl[t], s16x[t], s32[t])
end
m(v, r) = sum(v[r]) / length(r)
println("\nmean excess          t=1..8   9..11  12..16  17..32   mean loss 1..16  1..32")
for (name, v) in (("T8 run long", g), ("T8 sliding", sl), ("scratch16 run long", s16x), ("scratch32", s32))
    @printf("%-20s %6.3f  %6.3f  %6.3f  %6.3f   %6.4f  %6.4f\n", name, m(v, 1:8), m(v, 9:11), m(v, 12:16), m(v, 17:32),
            m(v .+ fl, 1:16), m(v .+ fl, 1:32))
end
@printf("%-20s                                  %6.4f  %6.4f\n", "floor", m(fl, 1:16), m(fl, 1:32))

# 3. staging: continue the T=8 model at T=32 and count tokens
if get(ENV, "STAGE", "1") == "1"
    θ = copy(models[8]); lg = Tuple{Int,Float64}[]
    train!(θ, AdamW(θ), data, 32, STEPS, MersenneTwister(99); every = 100,
           evalf = θ -> sum(excess(θ)) / Tmax, log = lg, work0 = STEPS * 32 * 8)
    println("\nmean excess at T=32 against tokens processed (thousands)")
    println("scratch32: ", join(["$(w ÷ 1000)k:$(round(e, digits = 3))" for (w, e) in logs[32][1:3:end]], "  "))
    println("staged 8→32 (work includes the T=8 stage): ", join(["$(w ÷ 1000)k:$(round(e, digits = 3))" for (w, e) in lg[1:3:end]], "  "))
    for thr in (0.2, 0.1, 0.05, 0.02)
        f(l) = (i = findfirst(x -> x[2] ≤ thr, l); i === nothing ? "never" : "$(l[i][1] ÷ 1000)k")
        println("tokens to mean excess ≤ $thr:  scratch32 ", f(logs[32]), "   staged ", f(lg),
                "   (staged, T=32 stage only: ", (i = findfirst(x -> x[2] ≤ thr, lg); i === nothing ? "never" : "$((lg[i][1] - STEPS*32*8) ÷ 1000)k"), ")")
    end
    ex = excess(θ)
    @printf("staged final mean excess: 1..8 %.3f  9..11 %.3f  12..16 %.3f  17..32 %.3f\n", m(ex, 1:8), m(ex, 9:11), m(ex, 12:16), m(ex, 17:32))
end
