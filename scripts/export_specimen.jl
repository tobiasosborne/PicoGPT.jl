# Export the specimen's complete trace for the JavaScript engine (BRIEF §5):
# its forward pass, gradient and training trajectory, every number in Float64.
#
#     julia --project scripts/export_specimen.jl        →  artifacts/specimen.json

include("common.jl")

const corpus = "to be or not to be"
const seed = 1
const hyper = (η = 2e-2, ηmin = 2e-3, warmup = 20, steps = 200,
               β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = 0.01, clip = 1.0)

abc = alphabet(corpus)
data = encode(corpus, abc)
c = Config(V = length(abc), D = 8, H = 2, L = 1, T = 8, F = 32)
x, y = windows(data, c.T)                       # every window of length T: B = 10, N = 80
B, N, d = size(x, 2), length(x), c.D ÷ c.H

# ── θ, laid out ──────────────────────────────────────────────────────────────────────
p = carve(zeros(nparams(c)), c)
names = ["E"; ["blocks[$ℓ].$k" for ℓ in 1:c.L for k in keys(p.blocks[ℓ])]; "γf"; "βf"; "Wout"; "bout"]
shortname(name) = Symbol(last(split(name, '.')))
pieces = leaves(p)
offsets = cumsum([0; length.(pieces)[1:end-1]])
layout = [(name, shape = collect(size(x)), offset) for (name, x, offset) in zip(names, pieces, offsets)]
initdist = [(name, dist = "normal", mean = μ, std = σ)
            for name in names for (μ, σ) in (initial(shortname(name), c),)]

θ₀ = init(c; rng = Xoshiro(seed))

# ── the forward pass, recorded, and the gradient ─────────────────────────────────────
trace = []
Z, back_Z = gpt(θ₀, x, c; trace)
ℓ, back_ℓ = crossentropy(Z, y)
θ̄ = back_Z(back_ℓ(1))
ℓ₂, back = loss(θ₀, x, y, c)
@assert ℓ₂ == ℓ && back() == θ̄

# S before the mask, computed here from Q and K (index form, independent of the model code)
scores(Q, K) = [dot(Q[(h-1)*d+1:h*d, (b-1)*c.T+i], K[(h-1)*d+1:h*d, (b-1)*c.T+j]) / √d
                for i in 1:c.T, j in 1:c.T, h in 1:c.H, b in 1:B]
M = [j > i ? -Inf : 0.0 for i in 1:c.T, j in 1:c.T]

blocks = map(trace[2:end-1]) do r
    S = scores(r.Q, r.K)
    @assert r.A ≈ mapslices(s -> softmax(s), S .+ M; dims = 2)      # A = softmax_rows(S + M)
    (ln1 = tensor(r.ln1), Q = tensor(r.Q), K = tensor(r.K), V = tensor(r.V), S = tensor(S),
     A = tensor(r.A), O = tensor(r.O), Xmid = tensor(r.Xmid), ln2 = tensor(r.ln2),
     U = tensor(r.U), R = tensor(r.R), Y = tensor(r.Y), Xout = tensor(r.Xout))
end
forward = (X0 = tensor(trace[1].X0), blocks, Xf = tensor(trace[end].Xf), Z = tensor(Z),
           probs = tensor(softmax(Z; dims = 1)), loss = ℓ)

# ── training: AdamW on the full batch, warmup + cosine, clipping ─────────────────────
function train(θ₀)
    θ = copy(θ₀)
    opt = AdamW(θ; hyper.β1, hyper.β2, hyper.ε, λ = decay(c, hyper.λ))
    η(t) = warmup_cosine(t; hyper.η, hyper.warmup, hyper.steps, hyper.ηmin)
    losses = [step!(θ, opt, x, y, c; η = η(t), hyper.clip) for t in 1:hyper.steps]
    push!(losses, first(loss(θ, x, y, c)))
    losses, θ
end
losses, θ = train(θ₀)                       # losses[k+1]: the loss after k updates
sample = decode(generate(θ, c, encode("to be or", abc), 10; τ = 0), abc)

# How much does rounding matter? Retrain from θ₀ perturbed by about one ulp per entry.
perturbed = [train(θ₀ .+ eps.(θ₀) .* randn(Xoshiro(k), length(θ₀))) for k in 1:3]
sensitivity = (loss = maximum(maximum(abs.(l .- losses)) for (l, _) in perturbed),
               theta_final = maximum(maximum(abs.(θ′ .- θ)) for (_, θ′) in perturbed))

# the entropy floor: the best possible loss, given that a context (window prefix) may be
# followed by different characters in different windows
followers = Dict{Vector{Int},Vector{Int}}()
for b in 1:B, t in 1:c.T
    push!(get!(followers, x[1:t, b], Int[]), y[t, b])
end
entropy_floor = sum(sum(-k * log(k / length(f)) for k in (count(==(a), f) for a in unique(f))) for f in values(followers)) / N

conventions = (
    arrays = "{shape, data}: data in column-major order, entry [i1, i2, …] (1-based) at " *
             "data[(i1-1) + shape[1]*((i2-1) + shape[2]*(…))] (0-based index into data)",
    tokens = "1-based ids into vocab; batch.x[b] is window b (its T input ids), batch.y[b] its T targets, y[t] = x[t+1]",
    columns = "token column n = (b-1)*T + t holds position t of window b; N = T*B",
    theta = "θ is one flat vector; layout gives each parameter's offset (0-based) and shape (column-major)",
    P = "P[2i-1, t] = sin(ω_i (t-1)), P[2i, t] = cos(ω_i (t-1)), ω_i = 10000^(-(i-1)/(D/2)), i = 1…D/2",
    X0 = "X0[:, n] = E[:, x_n] + P[:, t_n]",
    LN = "LN(x; γ, β) = γ ⊙ (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps) + β, per column",
    heads = "head h uses rows (h-1)d+1 … hd of Q, K, V, O, with d = D/H",
    S = "S[i, j, h, b] = q_i · k_j / √d for head h of window b, BEFORE the causal mask (j > i included)",
    A = "A[:, :, h, b] = softmax over each row of (S[:, :, h, b] + M), M[i, j] = -Inf for j > i, else 0",
    O = "O_h = V_h A_hᵀ per window, heads stacked back into D rows",
    block = "Xmid = X + Wo O + bo; ln2 = LN(Xmid; γ2, β2); U = W1 ln2 + b1; R = relu(U); Y = W2 R + b2; Xout = Xmid + Y",
    Xf = "Xf = LN(X_L; γf, βf), the input of the readout; Z = Wout Xf + bout",
    probs = "probs[:, n] = softmax(Z[:, n])",
    loss = "mean over the N columns of logsumexp(Z[:, n]) - Z[y_n, n]",
    grad = "∇θ of the loss at theta0, laid out like θ",
    train = "step t = 1…steps: g = ∇θ loss (full batch); if |g| > clip: g *= clip/|g|; then AdamW with η_t",
    schedule = "η_t = η t/warmup for t ≤ warmup, else ηmin + (η - ηmin)(1 + cos(π (t - warmup)/(steps - warmup)))/2",
    adamw = "m = β1 m + (1-β1) g; v = β2 v + (1-β2) g²; " *
            "θ = θ - η_t ((m/(1-β1^t)) / (sqrt(v/(1-β2^t)) + ε) + λ_i θ), m = v = 0 at the start",
    decay = "λ_i = weight_decay for entries of matrices (layout shapes with 2 dims), 0 for vectors (γ, β, biases)",
    trainloss = "train.loss[k+1] is the loss after k updates, k = 0 … steps (loss[1] = forward.loss up to rounding)",
)

specimen = (
    config = (V = c.V, D = c.D, H = c.H, F = c.F, L = c.L, T = c.T, eps = 1e-5),
    vocab = string.(abc),
    corpus,
    param_count = nparams(c),
    layout,
    init = (seed, rng = "Julia Xoshiro256++ (randn); reproduce the distributions, not the draws", params = initdist),
    P = tensor(positions(c.D, c.T)),
    batch = (x = collect.(eachcol(x)), y = collect.(eachcol(y))),
    theta0 = θ₀,
    forward,
    grad = θ̄,
    train = (optimizer = "adamw",
             hyper = (eta = hyper.η, eta_min = hyper.ηmin, warmup = hyper.warmup, schedule = "warmup_cosine",
                      beta1 = hyper.β1, beta2 = hyper.β2, eps = hyper.ε, weight_decay = hyper.λ,
                      decay_applies_to = "matrices", clip = hyper.clip, batch = "all windows, every step"),
             steps = hyper.steps, loss = losses, theta_final = θ,
             entropy_floor, greedy_from_to_be_or = sample,
             roundoff_sensitivity = (sensitivity...,
                 note = "largest change of the loss trajectory / of theta_final when θ₀ is perturbed " *
                        "by ~1 ulp per entry: the floor for comparing an independent implementation")),
    conventions,
)

path = writejson(artifact("specimen.json"), specimen)
@printf("specimen: p = %d parameters, B = %d windows, N = %d tokens\n", nparams(c), B, N)
@printf("loss at θ₀ %.6f (ln V = %.6f); after %d steps %.6f (entropy floor %.6f)\n",
        ℓ, log(c.V), hyper.steps, last(losses), entropy_floor)
println("greedy continuation of \"to be or\": ", repr(sample))
@printf("round-off sensitivity: loss trajectory %.1e, θ_final %.1e\n", sensitivity.loss, sensitivity.theta_final)
println("wrote ", path, " (", round(filesize(path) / 1024, digits = 1), " KiB)")
