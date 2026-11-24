# Check for grow/context: does the scratch T=32 baseline leave its plateau at another learning rate?
# Same world, model, init seed and data stream as context.jl with SEED=1; only η differs.
# From the repository root, about 5 min:
#   OPENBLAS_NUM_THREADS=1 julia --project -t1 grow/synthesis/context_lr.jl 1 5000 0.002,0.004,0.001
# Output (context_lr.txt): mean excess over the floor against tokens processed; η = 0.002
# reproduces the report's scratch curve (0.839 at 1024k, 0.692 at 2560k).
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
function perpos(θ, x, y, c)
    T, B = size(x); Z, _ = gpt(θ, x, c); s = zeros(T)
    for b in 1:B, t in 1:T
        z = @view Z[:, (b-1)*T+t]; s[t] += logsumexp(z) - z[y[t, b]]
    end
    s ./ B
end
cfg(T) = Config(; V, D = 32, H = 2, L = 1, T, F = 64)
data = stream(200_000, MersenneTwister(1)); test = stream(100_000, MersenneTwister(2))
xe, ye = windows(test, 32, 1:124:(length(test)-32)); fl = floorat.(1:32)
excess(θ) = sum(perpos(θ, xe, ye, cfg(32)) .- fl) / 32
SEED = parse(Int, ARGS[1]); steps = parse(Int, ARGS[2])
for η in parse.(Float64, split(ARGS[3], ","))
    θ = init(cfg(32); rng = MersenneTwister(10 + SEED)); opt = AdamW(θ); rng = MersenneTwister(20 + 32 + 1000SEED)
    out = String[]
    for s in 1:steps
        x, y = windows(data, 32, rand(rng, 1:length(data)-32, 32))
        step!(θ, opt, x, y, cfg(32); η = η * min(1, s / 50))
        s % 500 == 0 && push!(out, @sprintf("%dk:%.3f", s * 32 * 32 ÷ 1000, excess(θ)))
    end
    println("seed $SEED η=$η scratch T=32 mean excess: ", join(out, "  ")); flush(stdout)
end
