# From the repository root:
#   OPENBLAS_NUM_THREADS=1 julia -t2 --project grow/blocking-loss/run.jl
# Does the cheap block replacement (mean key, bias log|B|, mean value) cost language-model
# loss, and does it beat simply dropping the same distant positions?
#
# Identity used throughout. For query i and a block B of visible sources, one source with
# key mean(k_j), score bias log|B| and value mean(v_j) gets weight ∝ |B| exp(mean_j s_ij)
# and contributes that weight times mean(v_j). That is the same as giving every j ∈ B the
# averaged score mean_{j∈B} s_ij and running ordinary softmax and mix. So blocked attention
# is  A = softmax_rows(P_i S + M), with P_i the averaging projection over the groups of row i.
# P_i is symmetric, so the pullback is S̄ = P_i S̄'.  (Asserted numerically below.)
using PicoGPT, LinearAlgebra, Random, Printf, Statistics
BLAS.set_num_threads(1)

# ---------------------------------------------------------------- blocked attention
# G[i,j] for j ≤ i: 0 = source j dropped for query i; otherwise a group id in 1:T.
function group_mean!(out, s, G)
    T = size(s, 1); acc = zeros(T); cnt = zeros(Int, T)
    for i in 1:T
        fill!(acc, 0); fill!(cnt, 0)
        for j in 1:i
            g = G[i, j]; g > 0 && (acc[g] += s[i, j]; cnt[g] += 1)
        end
        for j in 1:T
            g = j ≤ i ? G[i, j] : 0
            out[i, j] = g > 0 ? acc[g] / cnt[g] : 0.0
        end
    end
    out
end

function battention(Q, K, V, H, T, labeler)
    D, N = size(Q); d = D ÷ H; B = N ÷ T; α = 1 / √d
    A = zeros(T, T, H, B); Gs = zeros(Int, T, T, H, B)
    sl(X, h, b) = view(X, (h-1)*d+1:h*d, (b-1)*T+1:b*T)
    hb = vec(collect(Iterators.product(1:H, 1:B)))
    Threads.@threads for (h, b) in hb
        q, k = sl(Q, h, b), sl(K, h, b)
        G = labeler(k, sl(V, h, b))
        Gs[:, :, h, b] .= G
        a = view(A, :, :, h, b)
        group_mean!(a, (q' * k) .* α, G)
        for i in 1:T, j in 1:T
            (j > i || G[i, j] == 0) && (a[i, j] = -Inf)
        end
        softmax!(a; dims = 2)
    end
    function back(Ā)
        Q̄, K̄ = zero(Q), zero(K)
        Threads.@threads for (h, b) in hb
            a = view(A, :, :, h, b)
            s̄ = a .* view(Ā, :, :, h, b)
            s̄ .-= a .* sum(s̄, dims = 2)
            t = group_mean!(similar(s̄), s̄, view(Gs, :, :, h, b))
            mul!(sl(Q̄, h, b), sl(K, h, b), t', α, false)
            mul!(sl(K̄, h, b), sl(Q, h, b), t, α, false)
        end
        Q̄, K̄
    end
    A, back, Gs
end

# src/block.jl and src/model.jl with attention replaced (the rules forbid editing src/).
function bblock(X₀, p, H, T, labeler)
    Â,  back_Â  = layernorm(X₀, p.γ1, p.β1)
    Q,  back_Q  = affine(Â, p.Wq, p.bq)
    K,  back_K  = affine(Â, p.Wk, p.bk)
    V,  back_V  = affine(Â, p.Wv, p.bv)
    A,  back_A, _ = battention(Q, K, V, H, T, labeler)
    O,  back_O  = mix(A, V)
    Y₁, back_Y₁ = affine(O, p.Wo, p.bo)
    X₁ = X₀ + Y₁
    F̂,  back_F̂  = layernorm(X₁, p.γ2, p.β2)
    U,  back_U  = affine(F̂, p.W1, p.b1)
    R,  back_R  = relu(U)
    Y₂, back_Y₂ = affine(R, p.W2, p.b2)
    X₂ = X₁ + Y₂
    function back(X̄₂)
        R̄, W̄2, b̄2   = back_Y₂(X̄₂)
        Ū            = back_R(R̄)
        F̂̄, W̄1, b̄1   = back_U(Ū)
        X̄₁, γ̄2, β̄2  = back_F̂(F̂̄)
        X̄₁ .+= X̄₂
        Ō, W̄o, b̄o   = back_Y₁(X̄₁)
        Ā, V̄         = back_O(Ō)
        Q̄, K̄         = back_A(Ā)
        Â̄q, W̄q, b̄q  = back_Q(Q̄)
        Â̄k, W̄k, b̄k  = back_K(K̄)
        Â̄v, W̄v, b̄v  = back_V(V̄)
        X̄₀, γ̄1, β̄1  = back_Â(Â̄q + Â̄k + Â̄v)
        X̄₀ .+= X̄₁
        X̄₀, (γ̄1, β̄1, W̄q, b̄q, W̄k, b̄k, W̄v, b̄v, W̄o, b̄o, γ̄2, β̄2, W̄1, b̄1, W̄2, b̄2)
    end
    X₂, back
end

function bgpt(θ, x, c, labeler)
    T = size(x, 1)
    p = carve(θ, c)
    X, back_embed = embed(x, p.E)
    backs = Vector{Any}(undef, c.L)
    for ℓ in 1:c.L
        X, backs[ℓ] = bblock(X, p.blocks[ℓ], c.H, T, labeler)
    end
    Xf, back_Xf = layernorm(X, p.γf, p.βf)
    Z, back_Z = affine(Xf, p.Wout, p.bout)
    function back(Z̄)
        X̄f, W̄out, b̄out = back_Z(Z̄)
        X̄, γ̄f, β̄f = back_Xf(X̄f)
        p̄ = Vector{Any}(undef, c.L)
        for ℓ in c.L:-1:1
            X̄, p̄[ℓ] = backs[ℓ](X̄)
        end
        flatten(back_embed(X̄), p̄, γ̄f, β̄f, W̄out, b̄out)
    end
    Z, back
end

function bloss(θ, x, y, c, labeler)
    Z, back_Z = bgpt(θ, x, c, labeler)
    ℓ, back_ℓ = crossentropy(Z, y)
    ℓ, () -> back_Z(back_ℓ(1))
end

# ---------------------------------------------------------------- which sources are kept
identity_labels(T) = [j for i in 1:T, j in 1:T]
static(G) = (k, v) -> G

"Blocks on the fixed grid of size b; a block is merged for query i only when it lies
entirely among the sources older than the r most recent (so it is complete: no partial block)."
function contiguous(T, b, r)
    G = identity_labels(T)
    for i in 1:T, j in 1:i
        first = ((j - 1) ÷ b) * b + 1
        first + b - 1 ≤ i - r && (G[i, j] = first)
    end
    G
end
"Drop exactly the positions that `contiguous` would merge."
drop_same(T, b, r) = (G = contiguous(T, b, r); I = identity_labels(T); [G[i, j] == I[i, j] && !(((j-1) ÷ b)*b + b ≤ i - r) ? j : 0 for i in 1:T, j in 1:T])
nsources(G, i) = length(unique(g for g in view(G, i, 1:i) if g > 0))
"Keep the n_i most recent positions, n_i = the number of sources G keeps for query i."
function drop_matched(G)
    T = size(G, 1)
    [j > i - nsources(G, i) ? j : 0 for i in 1:T, j in 1:T]
end
window(T, m) = [j > i - m ? j : 0 for i in 1:T, j in 1:T]

"Ward agglomeration of columns 1:m of F down to `target` clusters; ids = smallest member."
function ward!(cid, F, m, target)
    d = size(F, 1)
    μ = F[:, 1:m]; n = ones(m); alive = trues(m)
    for j in 1:m; cid[j] = j; end
    count = m
    while count > target
        best = Inf; ba = 0; bb = 0
        for a in 1:m
            alive[a] || continue
            for b in a+1:m
                alive[b] || continue
                s = 0.0
                for k in 1:d; s += (μ[k, a] - μ[k, b])^2; end
                cost = n[a] * n[b] / (n[a] + n[b]) * s
                cost < best && ((best, ba, bb) = (cost, a, b))
            end
        end
        for k in 1:d; μ[k, ba] = (n[ba] * μ[k, ba] + n[bb] * μ[k, bb]) / (n[ba] + n[bb]); end
        n[ba] += n[bb]; alive[bb] = false
        for j in 1:m; cid[j] == bb && (cid[j] = ba); end
        count -= 1
    end
end

"Same number of sources per query as contiguous(T, b, r), but the sources older than the r
most recent are grouped by smallest spread (Ward) of the projected keys, or of keys and values.
Only sources j ≤ i − r enter the clustering of row i, so the causal mask is respected."
function spread_labeler(T, b, r; values = false)
    Gc = contiguous(T, b, r)
    function (k, v)
        F = values ? vcat(k, v) : Matrix(k)
        G = identity_labels(T); cid = zeros(Int, T)
        for i in 1:T
            m = i - r
            m ≥ 2 || continue
            target = nsources(Gc, i) - r
            target < m || continue
            ward!(cid, F, m, target)
            G[i, 1:m] .= view(cid, 1:m)
        end
        G
    end
end

# ---------------------------------------------------------------- data, training, evaluation
const HERE = @__DIR__
const WORLD = get(ENV, "WORLD", "shakespeare")
if WORLD == "topic"
    # A world where evidence is spread over the whole context: the stream is cut into
    # segments of 96 tokens, each drawn i.i.d. from one of 8 hidden unigram distributions.
    # The best predictor averages over everything it has seen.
    wrng = Xoshiro(11); nV = 16
    dists = [(p = exp.(1.5 .* randn(wrng, nV)); cumsum(p ./ sum(p))) for _ in 1:8]
    function stream(n)
        s = Int[]
        while length(s) < n
            cdf = dists[rand(wrng, 1:8)]
            for _ in 1:96; push!(s, min(searchsortedfirst(cdf, rand(wrng)), nV)); end
        end
        s[1:n]
    end
    train, held = stream(54_000), stream(6_000)
else
    text = read(joinpath(HERE, "corpus.txt"), String)     # first 60,000 bytes of Tiny Shakespeare
    abc = alphabet(text); data = encode(collect(text), abc); nV = length(abc)
    train, held = data[1:54_000], data[54_001:end]
end
c = Config(; V = nV, D = 32, H = 4, L = 2, T = 32, F = 64)
T = c.T

function evaluate(θ, labeler; stride = 10)
    starts = 1:stride:length(held)-T
    nll = zeros(T); n = 0
    for J in Iterators.partition(starts, 128)
        x, y = windows(held, T, J)
        Z, _ = bgpt(θ, x, c, labeler)
        for (col, target) in enumerate(vec(y))
            nll[(col - 1) % T + 1] += logsumexp(view(Z, :, col)) - Z[target, col]
        end
        n += length(J)
    end
    nll ./ n                                           # held-out loss by position
end

"Per-window loss at positions 17-32, for paired comparisons."
function evaluate_windows(θ, labeler; stride = 10)
    per = Float64[]
    for J in Iterators.partition(1:stride:length(held)-T, 128)
        x, y = windows(held, T, J)
        Z, _ = bgpt(θ, x, c, labeler)
        for b in 1:length(J)
            push!(per, mean(logsumexp(view(Z, :, (b-1)*T + t)) - Z[y[t, b], (b-1)*T + t] for t in 17:T))
        end
    end
    per
end

function fit!(θ, labeler; steps, η, seed, warmup = 50)
    rng = Xoshiro(seed); opt = AdamW(θ; λ = decay(c, 0.01))
    for t in 1:steps
        x, y = windows(train, T, rand(rng, 1:length(train)-T, 32))
        ℓ, back = bloss(θ, x, y, c, labeler)
        update!(opt, θ, clip!(back(), 1), warmup_cosine(t; η, warmup, steps))
    end
    θ
end

late(v) = mean(v[17:end])
out = open(joinpath(HERE, WORLD == "shakespeare" ? "results.txt" : "results-$WORLD.txt"), "w")
say(fmt, args...) = (s = Printf.format(Printf.Format(fmt), args...); print(s); print(out, s); flush(out); flush(stdout))

# ---------------------------------------------------------------- checks
let rng = Xoshiro(7)
    # 1. explicit cheap replacement = score averaging
    d, m = 8, 5
    q = randn(rng, d); K = randn(rng, d, 9); V = randn(rng, d, 9); B = 3:7
    kB = vec(mean(K[:, B], dims = 2)); vB = vec(mean(V[:, B], dims = 2))
    rest = [1, 2, 8, 9]
    s = vcat(K[:, rest]' * q ./ √d, dot(q, kB) / √d + log(length(B)))
    w = softmax(s); explicit = hcat(V[:, rest], vB) * w
    s2 = K' * q ./ √d; s2[B] .= mean(s2[B]); averaged = V * softmax(s2)
    e1 = norm(explicit - averaged); @assert e1 < 1e-13
    # 2. with every source its own group, the copy reproduces PicoGPT's loss
    cs = Config(; V = 11, D = 8, H = 2, L = 2, T = 12, F = 16)
    θ = init(cs; rng); x = rand(rng, 1:11, 12, 3); y = rand(rng, 1:11, 12, 3)
    ℓ0, back0 = loss(θ, x, y, cs)
    ℓ1, back1 = bloss(θ, x, y, cs, static(identity_labels(12)))
    e2 = abs(ℓ0 - ℓ1); e3 = norm(back0() - back1()); @assert e2 < 1e-13 && e3 < 1e-12
    # 3. gradient of the blocked loss against central differences
    lab = static(contiguous(12, 3, 2))
    ℓ, back = bloss(θ, x, y, cs, lab); g = back(); e4 = 0.0
    for _ in 1:20
        u = randn(rng, length(θ)); u ./= norm(u); h = 1e-5
        fd = (bloss(θ + h*u, x, y, cs, lab)[1] - bloss(θ - h*u, x, y, cs, lab)[1]) / 2h
        e4 = max(e4, abs(fd - dot(g, u)) / max(abs(fd), 1e-8))
    end
    @assert e4 < 1e-6
    # 4. the spread labeler keeps exactly as many sources as the contiguous one, causally
    G = spread_labeler(12, 3, 2)(randn(rng, 4, 12), randn(rng, 4, 12)); Gc = contiguous(12, 3, 2)
    @assert all(nsources(G, i) == nsources(Gc, i) for i in 1:12)
    say("checks: cheap replacement vs score averaging %.1e | copy vs PicoGPT loss %.1e, gradient %.1e | blocked gradient vs finite differences (rel) %.1e\n", e1, e2, e3, e4)
end

# ---------------------------------------------------------------- 1. train
t0 = time()
θ = init(c; rng = Xoshiro(1))
fit!(θ, static(identity_labels(T)); steps = parse(Int, get(ENV, "STEPS", "1500")), η = 3e-3, seed = 2, warmup = 100)
base = evaluate(θ, static(identity_labels(T)))
say("world: %s\n", WORLD)
say("model V=%d D=%d H=%d L=%d T=%d F=%d, %d parameters; trained %.0f s on %d tokens; held out %d\n",
    c.V, c.D, c.H, c.L, c.T, c.F, nparams(c), time() - t0, length(train), length(held))
say("baseline held-out loss: all positions %.4f, positions 17-32 %.4f\n", mean(base), late(base))
say("baseline loss by position (= a truly shorter context of that length):\n  ")
for t in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32); say("t=%d: %.3f  ", t, base[t]); end
say("\n\n")

# ---------------------------------------------------------------- 2, 3. block against drop
srcs(G) = mean(nsources(G, i) for i in 17:T)
say("held-out loss at positions 17-32 (full attention %.4f). 'sources' = mean number of sources per query there (full: %.1f).\n", late(base), srcs(identity_labels(T)))
say("%-3s %-3s %8s | %10s %10s %10s\n", "b", "r", "sources", "blk-contig", "drop-same", "drop-match")
table = Dict()
grid = [(b, r) for r in (2, 4) for b in (2, 4, 8)]
for (b, r) in grid
    Gc = contiguous(T, b, r)
    res = (contig = late(evaluate(θ, static(Gc))),
           same = late(evaluate(θ, static(drop_same(T, b, r)))),
           matched = late(evaluate(θ, static(drop_matched(Gc)))))
    table[(b, r)] = res
    say("%-3d %-3d %8.2f | %10.4f %10.4f %10.4f\n", b, r, srcs(Gc), res...)
end
say("\npaired difference blk-contig − drop-match over %d held-out windows (mean ± standard error; windows overlap, so the error is optimistic):\n  ", length(1:10:length(held)-T))
for (b, r) in grid
    Gc = contiguous(T, b, r)
    δ = evaluate_windows(θ, static(Gc)) .- evaluate_windows(θ, static(drop_matched(Gc)))
    say("b=%d r=%d: %+.4f ± %.4f   ", b, r, mean(δ), std(δ) / √length(δ))
end
say("\n")
say("\nplain window of the m most recent sources in every layer:\n  ")
for m in (1, 2, 4, 8, 12, 16, 24); say("m=%d: %.4f  ", m, late(evaluate(θ, static(window(T, m))))); end
say("\n\n")

# ---------------------------------------------------------------- 4. train with the blocking in place
if get(ENV, "FINETUNE", "1") == "1"
    steps = parse(Int, get(ENV, "FT_STEPS", "250"))
    say("fine-tuning %d steps (η = 1e-3) from the trained model, with the modified attention in place; loss at positions 17-32:\n", steps)
    ctrl = fit!(copy(θ), static(identity_labels(T)); steps, η = 1e-3, seed = 3)
    say("  control, full attention, same extra steps: %.4f\n", late(evaluate(ctrl, static(identity_labels(T)))))
    say("%-3s %-3s %8s | %12s %12s | %12s %12s\n", "b", "r", "sources", "block before", "block after", "drop before", "drop after")
    for (b, r) in ((4, 4), (8, 2))
        Gc = contiguous(T, b, r); Gd = drop_matched(Gc)
        θb = fit!(copy(θ), static(Gc); steps, η = 1e-3, seed = 3)
        θd = fit!(copy(θ), static(Gd); steps, η = 1e-3, seed = 3)
        say("%-3d %-3d %8.2f | %12.4f %12.4f | %12.4f %12.4f\n", b, r, srcs(Gc), table[(b, r)].contig,
            late(evaluate(θb, static(Gc))), table[(b, r)].matched, late(evaluate(θd, static(Gd))))
    end
end

# ---------------------------------------------------------------- 2b. blocks chosen by smallest spread
say("\nblocks chosen by smallest spread (Ward), same number of sources per query as the contiguous blocks:\n")
say("%-3s %-3s %8s | %10s %10s %10s\n", "b", "r", "sources", "blk-contig", "blk-keys", "blk-key+val")
for (b, r) in grid
    say("%-3d %-3d %8.2f | %10.4f %10.4f %10.4f\n", b, r, srcs(contiguous(T, b, r)), table[(b, r)].contig,
        late(evaluate(θ, spread_labeler(T, b, r))), late(evaluate(θ, spread_labeler(T, b, r; values = true))))
end
say("total %.0f s\n", time() - t0)
close(out)
