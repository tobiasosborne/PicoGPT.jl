# Exact lifts of a PicoGPT model to a larger one, axis by axis.
#
# A lift takes (θ, net) to (θ′, net′) with net′ larger on one axis and, where the lift is
# exact, logits(θ′, x, net′) == logits(θ, x, net) to rounding on every input.
#
# The model is PicoGPT's (src/), used as a library and unchanged, with one generalisation:
# the positional code is P = U · positions(D₀, T) for a fixed matrix U (U = I is PicoGPT).
# That is what the width lift needs, because PicoGPT's code positions(D, T) changes with D.
#
# Notation (k copies): U = I_D ⊗ 1_k ∈ ℝ^{kD×D} duplicates coordinates, UᵀU = k I, and
# Uᵀ/k averages the copies. A = I − UUᵀ/k projects onto the antisymmetric part, A U = 0.

module Lifts

using PicoGPT, LinearAlgebra, Random
using PicoGPT: block, layernorm, affine, crossentropy, carve, flatten, initial

export Net, pos, logits, nll, lossgrad, dup, fresh,
       lift_width, lift_mlp, lift_depth, lift_context, lift_vocab, split_heads,
       symmetrise, stream_asymmetry, tangent, train!, mse

# ── the model, with the positional code as data ──────────────────────────────────────

"""
    Net(c, U)

A PicoGPT of shape `c` whose positional code is P = U · positions(D₀, T), D₀ = size(U, 2).
`Net(c)` is PicoGPT itself (U = I).
"""
struct Net
    c::Config
    U::Matrix{Float64}
end
Net(c::Config) = Net(c, Matrix{Float64}(I, c.D, c.D))
pos(net::Net, T) = net.U * positions(size(net.U, 2), T)

"""
    Z, back = logits(θ, x, net; trace = nothing)

PicoGPT's `gpt`, with the positional code of `net`. back(Z̄) = θ̄.
"""
function logits(θ, x, net::Net; trace = nothing)
    c = net.c
    T = size(x, 1)
    0 < T ≤ c.T || throw(ArgumentError("sequence length $T is not in 1:$(c.T)"))
    p = carve(θ, c)
    X = p.E[:, vec(x)]
    reshape(X, c.D, T, :) .+= pos(net, T)
    PicoGPT.record!(trace, (; X0 = X))
    backs = Vector{Any}(undef, c.L)
    for ℓ in 1:c.L
        X, backs[ℓ] = block(X, p.blocks[ℓ], c.H, T; trace)
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
        Ē = zero(p.E)                                  # the adjoint of a lookup: scatter-add
        for (n, a) in enumerate(x)
            @views Ē[:, a] .+= X̄[:, n]
        end
        flatten(Ē, p̄, γ̄f, β̄f, W̄out, b̄out)
    end
    Z, back
end

"""    ℓ, back = nll(θ, x, y, net)        mean surprise; back() = ∇ℓ"""
function nll(θ, x, y, net::Net)
    Z, back_Z = logits(θ, x, net)
    ℓ, back_ℓ = crossentropy(Z, y)
    ℓ, () -> back_Z(back_ℓ(1))
end

"""    ℓ, back = mse(θ, x, Z★, net)       mean squared logit error against Z★; back() = ∇ℓ"""
function mse(θ, x, Z★, net::Net)
    Z, back_Z = logits(θ, x, net)
    Δ = Z .- Z★
    sum(abs2, Δ) / length(Δ), () -> back_Z(2 .* Δ ./ length(Δ))
end

# ── helpers ──────────────────────────────────────────────────────────────────────────

"""    dup(n, k) = I_n ⊗ 1_k        coordinate i ↦ coordinates k(i−1)+1 … ki"""
dup(n, k) = kron(Matrix{Float64}(I, n, n), ones(k, 1))

"""A fresh draw of the parameter `name` at PicoGPT's initial scale for shape `c`."""
fresh(name, c, dims, rng) = ((μ, σ) = initial(name, c); μ .+ σ .* randn(rng, dims...))

config(c::Config; V = c.V, D = c.D, H = c.H, L = c.L, T = c.T, F = c.F) = Config(; V, D, H, L, T, F)

# ── width ────────────────────────────────────────────────────────────────────────────

"""
    θ′, net′ = lift_width(θ, net; k = 2, heads = :same, asym = 0, rng)

Width D → kD by duplicating coordinates. The stream becomes X′ = U X exactly, which
LayerNorm commutes with (same mean, same variance, ε included). Writers into the stream
are forced: W′ = U W. Readers from it need only W′ U = W, so

    W′ = W Uᵀ/k + asym · Ξ A,        Ξ a fresh random matrix, A = I − UUᵀ/k,

is exact for every `asym`: the second term reads the antisymmetric part of the stream,
which is zero at birth. `asym = 0` is plain duplication.

Heads:
  :same   H′ = H, d′ = kd. q′⋅k′ = s² k q⋅k and the score divides by √(kd), so the
          queries and keys carry s = k^(−1/4) each (one constant, k^(−1/2), in the scores).
  :more   H′ = kH, d′ = d. Every head is copied k times and Wo averages the copies.
          No constant is needed.
  :fresh  H′ = kH, d′ = d. The first copy is the parent's head; the others are new heads
          with random Wq, Wk, Wv and zero columns in Wo.

The positional code must be duplicated too: net′.U = U · net.U. PicoGPT's own code
positions(kD, T) is not of that form, so the child is not a PicoGPT `Config` alone.
"""
function lift_width(θ, net::Net; k = 2, heads = :same, asym = 0.0, rng = Random.default_rng())
    c = net.c
    (; D, H) = c
    d = D ÷ H
    c′ = config(c; D = k * D, H = heads == :same ? H : k * H)
    U = dup(D, k)
    A = I - U * U' / k
    # Uh: the parent's query/key/value rows ↦ the child's;  Uo: read back by Wo′ = U Wo Uoᵀ
    if heads == :same
        Uh, Uo, s = U, U / k, k^(-1 / 4)
    elseif heads == :more
        Uh = kron(Matrix{Float64}(I, H, H), kron(ones(k, 1), Matrix{Float64}(I, d, d)))
        Uo, s = Uh / k, 1.0
    elseif heads == :fresh
        e₁ = [j == 1 ? 1.0 : 0.0 for j in 1:k, _ in 1:1]
        Uh = kron(Matrix{Float64}(I, H, H), kron(e₁, Matrix{Float64}(I, d, d)))
        Uo, s = Uh, 1.0
    else
        throw(ArgumentError("heads must be :same, :more or :fresh"))
    end
    newrows = heads == :fresh ? Diagonal(1 .- vec(sum(Uh, dims = 2))) : 0.0 * I   # rows of the new heads
    reader(W, name) = W * U' / k + asym * fresh(name, c′, (size(W, 1), k * D), rng) * A
    headreader(W, name, s = 1.0) = s * Uh * W * U' / k + newrows * fresh(name, c′, (k * D, k * D), rng) +
                                   asym * fresh(name, c′, (k * D, k * D), rng) * A

    p = carve(θ, c)
    θ′ = zeros(nparams(c′))
    p′ = carve(θ′, c′)
    p′.E .= U * p.E
    for (b, b′) in zip(p.blocks, p′.blocks)
        b′.γ1 .= U * b.γ1;  b′.β1 .= U * b.β1
        b′.Wq .= headreader(b.Wq, :Wq, s);  b′.bq .= s * Uh * b.bq
        b′.Wk .= headreader(b.Wk, :Wk, s);  b′.bk .= s * Uh * b.bk
        b′.Wv .= headreader(b.Wv, :Wv);  b′.bv .= Uh * b.bv
        b′.Wo .= U * b.Wo * Uo';         b′.bo .= U * b.bo
        b′.γ2 .= U * b.γ2;  b′.β2 .= U * b.β2
        b′.W1 .= reader(b.W1, :W1);      b′.b1 .= b.b1
        b′.W2 .= U * b.W2;               b′.b2 .= U * b.b2
    end
    p′.γf .= U * p.γf;  p′.βf .= U * p.βf
    p′.Wout .= reader(p.Wout, :Wout);  p′.bout .= p.bout
    θ′, Net(c′, U * net.U)
end

"""
    θ′, net′, tail = split_heads(θ, net; m = 2)

Heads H → mH at fixed D (so d → d/m). NOT exact in general. Head h's scores are the
bilinear form x̂ᵢ ↦ [x̂ᵢ; 1]ᵀ M x̂ⱼ / √d, M = [Wq bq]ᵀ Wk of rank ≤ d (bk only shifts a row
of scores by a constant, which softmax ignores). A sub-head of width d/m carries a form
of rank ≤ d/m. This lift gives every sub-head the best rank-d/m approximation of M
(Eckart–Young) and leaves Wv, Wo alone; it is exact iff rank M ≤ d/m.
`tail` is, per block and head, the discarded fraction of ‖M‖_F.
"""
function split_heads(θ, net::Net; m = 2)
    c = net.c
    (; D, H) = c
    d = D ÷ H
    d % m == 0 || throw(ArgumentError("d = $d is not a multiple of m = $m"))
    r = d ÷ m
    c′ = config(c; H = m * H)
    θ′ = copy(θ)
    p, p′ = carve(θ, c), carve(θ′, c′)
    tail = Float64[]
    for (b, b′) in zip(p.blocks, p′.blocks), h in 1:H
        rows = (h-1)*d+1:h*d
        S = svd([b.Wq[rows, :] b.bq[rows]]' * b.Wk[rows, :])            # M, (D+1)×D
        push!(tail, norm(S.S[r+1:end]) / norm(S.S))
        root = m^(-1 / 4) .* sqrt.(S.S[1:r])                            # 1/√r = √m/√d
        Q̃, K̃ = root .* S.U[:, 1:r]', root .* S.Vt[1:r, :]
        for j in 1:m
            sub = (h-1)*d .+ ((j-1)*r+1:j*r)
            b′.Wq[sub, :] .= Q̃[:, 1:D];  b′.bq[sub] .= Q̃[:, D+1]
            b′.Wk[sub, :] .= K̃;          b′.bk[sub] .= 0
        end
    end
    θ′, Net(c′, net.U), tail
end

# ── MLP width ────────────────────────────────────────────────────────────────────────

"""
    θ′, net′ = lift_mlp(θ, net; F′, mode = :fresh, rng)

MLP width F → F′. All three are exact:
  :fresh  new hidden units read with random W1 rows and write with zero W2 columns;
  :dup    F′ = kF: every unit is copied k times and W2 averages the copies;
  :dead   new units have zero W1 rows and b1, and random W2 columns. Exact, but
          stillborn: relu(0) = 0 with backward value 0, so they never get a gradient.
"""
function lift_mlp(θ, net::Net; F′, mode = :fresh, rng = Random.default_rng())
    c = net.c
    (; D, F) = c
    F′ ≥ F || throw(ArgumentError("F′ must be at least F"))
    c′ = config(c; F = F′)
    p = carve(θ, c)
    θ′ = zeros(nparams(c′))
    p′ = carve(θ′, c′)
    p′.E .= p.E;  p′.γf .= p.γf;  p′.βf .= p.βf;  p′.Wout .= p.Wout;  p′.bout .= p.bout
    for (b, b′) in zip(p.blocks, p′.blocks)
        for name in (:γ1, :β1, :Wq, :bq, :Wk, :bk, :Wv, :bv, :Wo, :bo, :γ2, :β2, :b2)
            b′[name] .= b[name]
        end
        if mode == :dup
            F′ % F == 0 || throw(ArgumentError(":dup needs F′ a multiple of F"))
            Uf = dup(F, F′ ÷ F)
            b′.W1 .= Uf * b.W1;  b′.b1 .= Uf * b.b1;  b′.W2 .= b.W2 * Uf' / (F′ ÷ F)
        elseif mode == :fresh
            b′.W1 .= [b.W1; fresh(:W1, c′, (F′ - F, D), rng)]
            b′.b1 .= [b.b1; zeros(F′ - F)]
            b′.W2 .= [b.W2 zeros(D, F′ - F)]
        elseif mode == :dead
            b′.W1 .= [b.W1; zeros(F′ - F, D)]
            b′.b1 .= [b.b1; zeros(F′ - F)]
            b′.W2 .= [b.W2 fresh(:W2, c′, (D, F′ - F), rng)]
        else
            throw(ArgumentError("mode must be :fresh, :dup or :dead"))
        end
    end
    θ′, Net(c′, net.U)
end

# ── depth ────────────────────────────────────────────────────────────────────────────

"""
    θ′, net′ = lift_depth(θ, net; at = L + 1, rng)

A new block at position `at` that starts as the identity: Wo, bo, W2, b2 are zero, and
everything it reads with (Wq, Wk, Wv, W1, the gains) is a fresh random draw. No gate
parameter is needed; the two writing matrices are the gates.
"""
function lift_depth(θ, net::Net; at = net.c.L + 1, rng = Random.default_rng())
    c = net.c
    1 ≤ at ≤ c.L + 1 || throw(ArgumentError("at must be in 1:L+1"))
    c′ = config(c; L = c.L + 1)
    p = carve(θ, c)
    θ′ = zeros(nparams(c′))
    p′ = carve(θ′, c′)
    p′.E .= p.E;  p′.γf .= p.γf;  p′.βf .= p.βf;  p′.Wout .= p.Wout;  p′.bout .= p.bout
    for ℓ′ in 1:c′.L, name in keys(p′.blocks[ℓ′])
        x′ = p′.blocks[ℓ′][name]
        if ℓ′ == at
            x′ .= name in (:Wo, :bo, :W2, :b2) ? 0.0 : fresh(name, c′, size(x′), rng)
        else
            x′ .= p.blocks[ℓ′ < at ? ℓ′ : ℓ′ - 1][name]
        end
    end
    θ′, Net(c′, net.U)
end

# ── context length ───────────────────────────────────────────────────────────────────

"""
    θ′, net′ = lift_context(θ, net; T′)

Context T → T′. No parameter depends on T and column t of the positional code depends
only on t, so the lift is the identity on θ. On a longer sequence the child's logits at
positions t ≤ T are the parent's on the prefix (causality); positions t > T are new.
"""
lift_context(θ, net::Net; T′) = (copy(θ), Net(config(net.c; T = T′), net.U))

# ── vocabulary ───────────────────────────────────────────────────────────────────────

"""
    θ′, net′, π = lift_vocab(θ, net; v, m, q = fill(1/m, m))

Split token v into m tokens (v, 1) … (v, m): (v, 1) keeps the number v, the others are
V+1 … V+m−1. `π` maps a child token to its parent. Every copy gets v's embedding column
and readout row, and bout′[(v, b)] = bout[v] + log q_b. Then

    z′[(v, b)] = z[v] + log q_b,     z′[u] = z[u] for u ≠ v,

so Σ_b p′((v, b) | h′) = p(v | π h′) and p′(u | h′) = p(u | π h′): the child predicts
p(v | h) · q_b. The partition function is unchanged.
"""
function lift_vocab(θ, net::Net; v, m, q = fill(1 / m, m))
    c = net.c
    sum(q) ≈ 1 && length(q) == m || throw(ArgumentError("q must be a distribution on m values"))
    π = [1:c.V; fill(v, m - 1)]
    c′ = config(c; V = c.V + m - 1)
    p = carve(θ, c)
    θ′ = zeros(nparams(c′))
    p′ = carve(θ′, c′)
    p′.E .= p.E[:, π]
    foreach((b, b′) -> foreach(name -> b′[name] .= b[name], keys(b)), p.blocks, p′.blocks)
    p′.γf .= p.γf;  p′.βf .= p.βf
    p′.Wout .= p.Wout[π, :]
    p′.bout .= p.bout[π]
    p′.bout[[v; c.V+1:c.V+m-1]] .+= log.(q)
    θ′, Net(c′, net.U), π
end

# ── the symmetric and the new directions ─────────────────────────────────────────────

"""
    symmetrise(θ′, net′, U)

Project θ′ onto the fixed subspace of the copy-swap symmetry of a width lift with
duplication matrix U: average every stream index over its copies (rows of E and of the
writers Wo, W2, the stream vectors γ, β, bo, b2; columns of the readers Wq, Wk, Wv, W1,
Wout). The loss is invariant under permuting the copies, provided the positional code
is, so the gradient at a fixed point is fixed too.
"""
function symmetrise(θ′, net′::Net, U)
    k = size(U, 1) ÷ size(U, 2)
    S = U * U' / k
    θ = copy(θ′)
    p = carve(θ, net′.c)
    p.E .= S * p.E
    for b in p.blocks
        for name in (:γ1, :β1, :Wo, :bo, :γ2, :β2, :W2, :b2)
            b[name] .= S * b[name]
        end
        for name in (:Wq, :Wk, :Wv, :W1)
            b[name] .= b[name] * S
        end
    end
    p.γf .= S * p.γf;  p.βf .= S * p.βf
    p.Wout .= p.Wout * S
    θ
end

"""
    stream_asymmetry(θ′, x, net′, U) -> (X0, Xmid, Xout)

The fraction of the residual stream's norm outside the duplicated subspace range(U), at
the embedding, after attention and after the MLP of the last block.
"""
function stream_asymmetry(θ′, x, net′::Net, U)
    k = size(U, 1) ÷ size(U, 2)
    A = I - U * U' / k
    trace = []
    logits(θ′, x, net′; trace)
    frac(X) = norm(A * X) / norm(X)
    X0 = trace[1].X0
    last_block = trace[end]
    (X0 = frac(X0), Xmid = frac(last_block.Xmid), Xout = frac(last_block.Xout))
end

"""
    J, θ₀′ = tangent(lift, n)

A lift is affine in the parent's parameters: θ′ = θ₀′ + J θ. `lift(θ)` must be
deterministic (fix its random numbers). The columns of J span the parent's directions
in the child's parameter space.
"""
function tangent(lift, n)
    θ₀′ = lift(zeros(n))
    J = zeros(length(θ₀′), n)
    e = zeros(n)
    for i in 1:n
        e[i] = 1
        J[:, i] .= lift(e) .- θ₀′
        e[i] = 0
    end
    J, θ₀′
end

# ── training ─────────────────────────────────────────────────────────────────────────

"""
    losses = train!(θ, net, x, y; steps, η, warmup = 20, λ = 0.01, clip = 1,
                    target = -Inf, basis = nothing, each = nothing)

AdamW on the full batch with a linear warm-up to a constant step η, gradient clipping
and decoupled weight decay on the matrices: PicoGPT's recipe for the specimen without
the cosine. `losses[t]` is the loss after t − 1 updates; training stops as soon as
that is ≤ `target`, so `length(losses) − 1` updates were made.

With `basis = N` (orthonormal columns), only θ = θ₀ + N φ moves: AdamW runs on φ with
gradient Nᵀθ̄ and decay Nᵀ(λ ⊙ θ). `each(t, θ)` is called before every update.
"""
function train!(θ, net::Net, x, y; steps, η, warmup = 20, λ = 0.01, clip = 1.0,
                target = -Inf, basis = nothing, each = nothing, β1 = 0.9, β2 = 0.95, ε = 1e-8)
    Λ = decay(net.c, λ)
    n = basis === nothing ? length(θ) : size(basis, 2)
    m, v = zeros(n), zeros(n)
    losses = Float64[]
    for t in 1:steps+1
        ℓ, back = nll(θ, x, y, net)
        push!(losses, ℓ)
        each === nothing || each(t - 1, θ)
        (ℓ ≤ target || t > steps) && break
        g = clip!(back(), clip)
        w = Λ .* θ
        if basis !== nothing
            g, w = basis' * g, basis' * w
        end
        ηₜ = η * min(1, t / max(warmup, 1))
        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g^2
        Δ = @. ηₜ * ((m / (1 - β1^t)) / (√(v / (1 - β2^t)) + ε) + w)
        θ .-= basis === nothing ? Δ : basis * Δ
    end
    losses
end

end # module
