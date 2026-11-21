module WindowLoop

using PicoGPT, LinearAlgebra
using PicoGPT: foreach_head, record!

export window_mask, masked_attention, windowgpt, windowloss, register_window, register_windowloss

"""An additive causal mask whose window includes the query itself."""
function window_mask(T, width = 2, R = Float64)
    width ≥ 1 || throw(ArgumentError("window width must be positive"))
    R[0 ≤ i - j < width ? 0 : -Inf for i in 1:T, j in 1:T]
end

"""PicoGPT attention with a supplied fixed additive mask, shared by heads and words."""
function masked_attention(Q, K, H, M::AbstractMatrix)
    D, N = size(Q)
    T = size(M, 1)
    size(M) == (T, T) && T > 0 && N % T == 0 ||
        throw(DimensionMismatch("the mask must be square and match the sequence length"))
    size(K) == size(Q) && H > 0 && D % H == 0 ||
        throw(DimensionMismatch("query and key shapes must agree and divide into heads"))
    A = similar(Q, T, T, H, N ÷ T)
    α = inv(sqrt(eltype(Q)(D ÷ H)))
    foreach_head(A, Q, K) do a, q, k
        mul!(a, q', k, α, false)
        softmax!(a .+= M; dims = 2)
    end
    function back(Ā)
        S̄, Q̄, K̄ = similar(A), similar(Q), similar(K)
        foreach_head(A, Ā, S̄, Q, K, Q̄, K̄) do a, ā, s̄, q, k, q̄, k̄
            s̄ .= a .* ā
            s̄ .-= a .* sum(s̄, dims = 2)
            mul!(q̄, k, s̄', α, false)
            mul!(k̄, q, s̄, α, false)
        end
        Q̄, K̄
    end
    A, back
end

"""
The width-two mask followed by value mixing. Store just the self and previous
weights at each query. The first token has a singleton softmax with derivative zero.
A fixed logit bias can give the predecessor a prior without changing the learned
query/key score or its pullback.
"""
function window_mix(Q, K, V, H, T; lag_bias = 0)
    D, N = size(Q)
    d = D ÷ H
    α = inv(sqrt(eltype(Q)(d)))
    bias = eltype(Q)(lag_bias)
    weights = similar(Q, 2, H, N)
    O = similar(V)
    @inbounds for n in 1:N, h in 1:H
        rows = (h - 1) * d + 1:h * d
        if mod(n - 1, T) == 0
            weights[1, h, n], weights[2, h, n] = 1, 0
            for i in rows
                O[i, n] = V[i, n]
            end
            continue
        end
        gap = zero(eltype(Q))
        for i in rows
            gap += Q[i, n] * (K[i, n] - K[i, n-1])
        end
        gap = gap * α - bias
        e = exp(-abs(gap))
        high, low = inv(1 + e), e / (1 + e)
        a, b = gap ≥ 0 ? (high, low) : (low, high)
        weights[1, h, n], weights[2, h, n] = a, b
        for i in rows
            O[i, n] = a * V[i, n] + b * V[i, n-1]
        end
    end
    function back(Ō)
        Q̄, K̄, V̄ = zero(Q), zero(K), zero(V)
        @inbounds for n in 1:N, h in 1:H
            rows = (h - 1) * d + 1:h * d
            if mod(n - 1, T) == 0
                for i in rows
                    V̄[i, n] = Ō[i, n]
                end
                continue
            end
            a, b = weights[1, h, n], weights[2, h, n]
            gap̄ = zero(eltype(Q))
            for i in rows
                gap̄ += Ō[i, n] * (V[i, n] - V[i, n-1])
                V̄[i, n] += a * Ō[i, n]
                V̄[i, n-1] += b * Ō[i, n]
            end
            gap̄ *= α * a * b
            for i in rows
                Q̄[i, n] = gap̄ * (K[i, n] - K[i, n-1])
                K̄[i, n] += gap̄ * Q[i, n]
                K̄[i, n-1] -= gap̄ * Q[i, n]
            end
        end
        Q̄, K̄, V̄
    end
    O, back, weights
end

function window_block(X, p, H, T; correction_only = false, anchor = nothing, lag_bias = 0)
    Â, back_ln1 = layernorm(X, p.γ1, p.β1)
    Q, back_q = affine(Â, p.Wq, p.bq)
    K, back_k = affine(Â, p.Wk, p.bk)
    V, back_v = affine(Â, p.Wv, p.bv)
    O, back_mix, weights = window_mix(Q, K, V, H, T; lag_bias)
    Y₁, back_o = affine(O, p.Wo, p.bo)
    X₁ = (isnothing(anchor) ? X : anchor) + Y₁
    F̂, back_ln2 = layernorm(X₁, p.γ2, p.β2)
    U, back_1 = affine(F̂, p.W1, p.b1)
    R, back_relu = relu(U)
    Y₂, back_2 = affine(R, p.W2, p.b2)
    function back(X̄₂)
        R̄, W̄2, b̄2 = back_2(X̄₂)
        F̂̄, W̄1, b̄1 = back_1(back_relu(R̄))
        X̄₁, γ̄2, β̄2 = back_ln2(F̂̄)
        X̄₁ .+= X̄₂
        Ō, W̄o, b̄o = back_o(X̄₁)
        Q̄, K̄, V̄ = back_mix(Ō)
        Āq, W̄q, b̄q = back_q(Q̄)
        Āk, W̄k, b̄k = back_k(K̄)
        Āv, W̄v, b̄v = back_v(V̄)
        X̄, γ̄1, β̄1 = back_ln1(Āq + Āk + Āv)
        if isnothing(anchor)
            X̄ .+= correction_only ? X̄₁ - X̄₂ : X̄₁
        end
        d_block = (γ̄1, β̄1, W̄q, b̄q, W̄k, b̄k, W̄v, b̄v, W̄o, b̄o, γ̄2, β̄2, W̄1, b̄1, W̄2, b̄2)
        d_anchor = isnothing(anchor) ? nothing : X̄₁
        X̄, d_block, d_anchor
    end
    (correction_only ? Y₁ + Y₂ : X₁ + Y₂), back, weights
end

function readout(X, p)
    Xf, back_norm = layernorm(X, p.γf, p.βf)
    Z, back_affine = affine(Xf, p.Wout, p.bout)
    function back(Z̄)
        X̄f, W̄out, b̄out = back_affine(Z̄)
        X̄, γ̄f, β̄f = back_norm(X̄f)
        X̄, (γ̄f, β̄f, W̄out, b̄out)
    end
    Z, back
end

function window_embed(x, E, position)
    position == :sinusoidal && return embed(x, E)
    position in (:none, :boundary) || throw(ArgumentError("unknown position mode"))
    T = size(x, 1)
    T > 0 || throw(ArgumentError("sequences must contain at least one token"))
    X = E[:, vec(x)]
    if position == :boundary
        # Break translation symmetry at the first token only.
        for n in 1:T:length(x)
            @views X[1:2:end, n] .+= 1
            @views X[2:2:end, n] .-= 1
        end
    end
    function back(X̄)
        Ē = zero(E)
        for (n, a) in enumerate(x)
            @views Ē[:, a] .+= X̄[:, n]
        end
        Ē
    end
    X, back
end

# BOS has an input embedding only. Its column participates in the recurrent
# stream, but its logits are neither returned nor scored.
function window_start(θ, x, c, position)
    T, B = size(x)
    T > 0 || throw(ArgumentError("sequences must contain at least one token"))
    if position != :bos
        p = carve(θ, c)
        X₀, back_embed = window_embed(x, p.E, position)
        return p, X₀, back_embed, T, :, length(θ)
    end
    P = nparams(c)
    length(θ) == P + c.D || throw(DimensionMismatch("missing input-only BOS embedding"))
    p = carve(view(θ, 1:P), c)
    bos = view(θ, P+1:length(θ))
    padded = vcat(fill(c.V + 1, 1, B), x)
    X₀, back_embed = window_embed(padded, hcat(p.E, bos), :none)
    data = [n for n in 1:length(padded) if mod(n - 1, T + 1) != 0]
    p, X₀, back_embed, T + 1, data, P
end

data_stream(X, ::Colon) = X
data_stream(X, data) = X[:, data]

function add_readout_gradient!(X̄, ::Colon, dX)
    X̄ .+= dX
end

function add_readout_gradient!(X̄, data, dX)
    view(X̄, :, data) .+= dX
end

function initial_stream(X₀, T, start)
    start == :embedded && return X₀
    start == :boundary || throw(ArgumentError("unknown initial stream"))
    X = zero(X₀)
    @views X[:, 1:T:end] .= X₀[:, 1:T:end]
    X
end

function advance(X, X₀, p, H, T, injection, lag_bias)
    anchored = injection in (:anchor, :anchor_normalized)
    Y, back, weights = window_block(X, p, H, T; correction_only = injection == :refresh,
                                    anchor = anchored ? X₀ : nothing, lag_bias)
    (anchored || injection == :none) || (Y = Y + X₀)
    if injection in (:normalized, :anchor_normalized)
        R, D = eltype(Y), size(Y, 1)
        Y, back_norm = layernorm(Y, ones(R, D), zeros(R, D))
        return Y, back, weights, Δ -> first(back_norm(Δ))
    end
    Y, back, weights, identity
end

function check_window(c, K, tied, injection)
    K ≥ 1 && (tied ? c.L == 1 : c.L ≥ K) ||
        throw(ArgumentError("use L=1 for tied passes or L≥K for untied passes, with K ≥ 1"))
    injection in (:after, :none, :refresh, :normalized, :anchor, :anchor_normalized) ||
        throw(ArgumentError("unknown injection mode"))
end

"""Visit each forward pass without retaining the unrolled pullback."""
function visit_passes(f, θ, x, c::Config; K, tied = true, injection = :after,
                      position = :sinusoidal, lag_bias = 0, start = :embedded)
    check_window(c, K, tied, injection)
    p, X₀, _, T, data, _ = window_start(θ, x, c, position)
    X = initial_stream(X₀, T, start)
    for k in 1:K
        previous = X
        X, _, weights, _ = advance(X, X₀, p.blocks[tied ? 1 : k], c.H, T, injection, lag_bias)
        Z, _ = readout(data_stream(X, data), p)
        if position == :bos
            f(k, Z, data_stream(X, data), data_stream(previous, data), weights[:, :, data])
        else
            f(k, Z, X, previous, weights)
        end
    end
end

"""
Run tied or untied width-two blocks with stock embeddings and readout.
An optional forward trace records streams and the two weights at every query
(self first). The pullback can record state cotangents in reverse pass order.
With `position = :bos`, append the input-only BOS embedding to the parameter vector.
The trace includes BOS columns, while the returned logits contain only data columns.
"""
function windowgpt(θ, x, c::Config; K::Int = c.L, tied::Bool = true, injection = :after,
                   position = :sinusoidal, lag_bias = 0, start = :embedded,
                   readouts = :last, trace = nothing)
    check_window(c, K, tied, injection)
    readouts in (:all, :last) || throw(ArgumentError("unknown readout mode"))

    p, X₀, back_embed, T, data, P = window_start(θ, x, c, position)
    X = initial_stream(X₀, T, start)
    record!(trace, (; pass = 0, X))
    Zs = typeof(X₀)[]
    back_blocks = Vector{Function}(undef, K)
    back_streams = Vector{Function}(undef, K)
    back_readouts = Vector{Function}(undef, readouts == :all ? K : 1)
    for k in 1:K
        X, back_blocks[k], weights, back_streams[k] = advance(
            X, X₀, p.blocks[tied ? 1 : k], c.H, T, injection, lag_bias)
        if readouts == :all || k == K
            Z, back_Z = readout(data_stream(X, data), p)
            push!(Zs, Z)
            back_readouts[length(Zs)] = back_Z
        end
        record!(trace, (; pass = k, X, weights))
    end

    function back(Z̄; adjoints = nothing)
        X̄, X̄₀, θ̄ = zero(X₀), zero(X₀), zero(θ)
        q = carve(view(θ̄, 1:P), c)
        for k in K:-1:1
            if readouts == :all || k == K
                dX, d_readout = readouts == :all ? back_readouts[k](Z̄[k]) : back_readouts[1](Z̄)
                add_readout_gradient!(X̄, data, dX)
                for (dst, src) in zip((q.γf, q.βf, q.Wout, q.bout), d_readout)
                    dst .+= src
                end
            end
            isnothing(adjoints) || record!(adjoints, (; pass = k, X̄ = copy(X̄)))
            Δ = back_streams[k](X̄)
            X̄, d_block, d_anchor = back_blocks[k](Δ)
            if !isnothing(d_anchor)
                X̄₀ .+= d_anchor
            elseif injection != :none
                X̄₀ .+= Δ
            end
            for (dst, src) in zip(values(q.blocks[tied ? 1 : k]), d_block)
                dst .+= src
            end
        end
        record!(adjoints, (; pass = 0, X̄))
        if start == :boundary
            @views X̄₀[:, 1:T:end] .+= X̄[:, 1:T:end]
        else
            X̄₀ .+= X̄
        end
        Ē = back_embed(X̄₀)
        q.E .= view(Ē, :, 1:c.V)
        if position == :bos
            θ̄[P+1:end] .= view(Ē, :, c.V+1)
        end
        θ̄
    end
    (readouts == :all ? Zs : only(Zs)), back
end

"""Ordinary prefix cross-entropy or its exact conditional expectation outside the cone."""
function prefix_term(Z, y, front)
    front ≥ size(y, 1) && return crossentropy(Z, y)
    # The target probability of class 2 is exactly 1/2 when an unseen fair
    # bit remains. A binary logit gap suffices for the loss and pullback.
    R, T, N = eltype(Z), size(y, 1), length(y)
    gradient = similar(Z)
    loss = zero(R)
    @inbounds for n in 1:N
        target = mod(n - 1, T) < front ? R(y[n] - 1) : R(0.5)
        gap = Z[2, n] - Z[1, n]
        e = exp(-abs(gap))
        p = (gap ≥ 0 ? one(R) : e) / (1 + e)
        loss += log1p(e) + (gap ≥ 0 ? (1 - target) * gap : -target * gap)
        gradient[2, n] = (p - target) / N
        gradient[1, n] = -gradient[2, n]
    end
    loss / N, Δ -> R(Δ) .* gradient
end

function windowloss(θ, x, y, c::Config; K = c.L, supervision = :deep,
                    conditional = false, opts...)
    supervised_window(windowgpt, θ, x, y, c; K, supervision, conditional, opts...)
end

function supervised_window(forward, θ, x, y, c; K, supervision, conditional, opts...)
    supervision in (:deep, :last) || throw(ArgumentError("unknown supervision mode"))
    conditional && c.V != 2 && throw(ArgumentError("conditioned prefix parity needs two symbols"))
    isempty(x) && throw(ArgumentError("the mean loss needs a nonempty batch"))
    if supervision == :last
        size(y) == size(x) || throw(DimensionMismatch("inputs and targets must have the same shape"))
        Z, back_Z = forward(θ, x, c; K, readouts = :last, opts...)
        ℓ, back_ℓ = prefix_term(Z, y, conditional ? K + 1 : size(y, 1))
        return ℓ, (Δ = 1) -> back_Z(back_ℓ(Δ))
    end
    ys = y isa AbstractMatrix ? fill(y, K) : y
    length(ys) == K && all(yk -> size(yk) == size(x), ys) ||
        throw(DimensionMismatch("deep supervision needs a target matrix at every pass"))
    Zs, back_Z = forward(θ, x, c; K, readouts = :all, opts...)
    terms = map(1:K) do k
        yk = ys[k]
        prefix_term(Zs[k], yk, conditional ? k + 1 : size(yk, 1))
    end
    back(Δ = 1) = back_Z([pull(Δ / K) for (_, pull) in terms])
    sum(first, terms) / K, back
end

register_stride(c, transition) = transition == :gain ? 2 : 4c.F + 1

function register_parameters(θ, c, transition)
    stride = register_stride(c, transition)
    length(θ) == stride * c.L || throw(DimensionMismatch("wrong register parameter count"))
    [view(θ, (k - 1) * stride + 1:k * stride) for k in 1:c.L]
end

function register_mlp(p, F)
    (; W1 = reshape(view(p, 1:2F), F, 2), b1 = view(p, 2F+1:3F),
       W2 = reshape(view(p, 3F+1:4F), 1, F), b2 = view(p, 4F+1:4F+1))
end

function register_cell(p, pair, F, transition)
    if transition == :gain
        source, left = view(pair, 1:1, :), view(pair, 2:2, :)
        a, b = (1 .- source) ./ 2, (1 .+ source) ./ 2
        gain = p[1] .* a .+ p[2] .* b
        function back_gain(ū)
            pair̄ = vcat(((p[2] - p[1]) / 2) .* left .* ū, gain .* ū)
            pair̄, [sum(ū .* left .* a), sum(ū .* left .* b)]
        end
        return gain .* left, back_gain
    end
    (; W1, b1, W2, b2) = register_mlp(p, F)
    U, back_1 = affine(pair, W1, b1)
    H, back_relu = relu(U)
    u, back_2 = affine(H, W2, b2)
    function back_mlp(ū)
        H̄, W̄2, b̄2 = back_2(ū)
        pair̄, W̄1, b̄1 = back_1(back_relu(H̄))
        pair̄, vcat(vec(W̄1), b̄1, vec(W̄2), b̄2)
    end
    u, back_mlp
end

function register_start(x, R, task)
    T = size(x, 1)
    T > 0 || throw(ArgumentError("sequences must contain at least one token"))
    source = reshape(R.(2 .* x .- 3), 1, :)
    task == :rule60 && return source, source
    state = zero(source)
    @views state[1, 1:T:end] .= source[1, 1:T:end]
    source, state
end

function register_step(state, source, p, F, T, task, transition)
    N = size(state, 2)
    pair = fill(-one(eltype(state)), 2, N)
    @views pair[1, :] .= vec(task == :prefix ? source : state)
    for n in 2:N
        mod(n - 1, T) == 0 || (pair[2, n] = state[1, n-1])
    end
    u, back_cell = register_cell(p, pair, F, transition)
    next = tanh.(u)
    function back(ū)
        pair̄, p̄ = back_cell(ū)
        statē = task == :prefix ? zero(state) : pair̄[1:1, :]
        for n in 2:N
            mod(n - 1, T) == 0 || (statē[1, n-1] += pair̄[2, n])
        end
        statē, p̄
    end
    next, vcat(zero(u), u), back
end

function check_register(c, K, tied, task, transition)
    check_window(c, K, tied, :none)
    c.V == 2 && c.H == 1 || throw(ArgumentError("the register uses two symbols and one head"))
    task in (:prefix, :rule60) || throw(ArgumentError("unknown register task"))
    transition in (:gain, :mlp) || throw(ArgumentError("unknown register transition"))
end

function visit_register(f, θ, x, c; K, tied = true, task = :prefix, transition = :gain)
    check_register(c, K, tied, task, transition)
    T = size(x, 1)
    source, state = register_start(x, eltype(θ), task)
    params = register_parameters(θ, c, transition)
    weights = ones(eltype(θ), 2, 1, length(x))
    @views weights[1, :, :] .= 0
    for k in 1:K
        previous = state
        state, Z, _ = register_step(state, source, params[tied ? 1 : k], c.F, T, task, transition)
        # An isometric, column-centred representation of the scalar register.
        f(k, Z, vcat(-state, state) ./ sqrt(eltype(state)(2)),
          vcat(-previous, previous) ./ sqrt(eltype(state)(2)), weights)
    end
end

"""
A bounded scalar state with a fixed predecessor head and a virtual left bit of zero.
For prefix parity, the head reads the provisional parity and the transition sees the
immutable current symbol. At pass zero only the first site's parity is known and the
remaining states are zero log-odds. For Rule 60 the transition reads the current state
and the state starts with every input bit. The transition is either a two-input ReLU
MLP or two learned signed gains interpolated by the current symbol/state.
"""
function register_window(θ, x, c::Config; K = c.L, tied = true, task = :prefix,
                         transition = :gain, readouts = :last)
    check_register(c, K, tied, task, transition)
    readouts in (:all, :last) || throw(ArgumentError("unknown readout mode"))
    T = size(x, 1)
    source, state = register_start(x, eltype(θ), task)
    params = register_parameters(θ, c, transition)
    Zs = typeof(state)[]
    back_steps = Vector{Function}(undef, K)
    states = Vector{typeof(state)}(undef, K)
    for k in 1:K
        state, Z, back_steps[k] = register_step(state, source, params[tied ? 1 : k],
                                               c.F, T, task, transition)
        states[k] = state
        (readouts == :all || k == K) && push!(Zs, Z)
    end
    function back(Z̄; adjoints = nothing)
        θ̄ = zero(θ)
        gradients = register_parameters(θ̄, c, transition)
        statē = zero(source)
        zero_direct = zero(source)
        for k in K:-1:1
            direct = readouts == :all ? Z̄[k][2:2, :] : k == K ? Z̄[2:2, :] : zero_direct
            # Record the pre-tanh stream cotangent, where the logits are read.
            ū = direct .+ statē .* (1 .- states[k].^2)
            record!(adjoints, (; pass = k, X̄ = ū))
            statē, p̄ = back_steps[k](ū)
            gradients[tied ? 1 : k] .+= p̄
        end
        record!(adjoints, (; pass = 0, X̄ = statē))
        θ̄
    end
    (readouts == :all ? Zs : only(Zs)), back
end

register_windowloss(θ, x, y, c; K = c.L, supervision = :deep, conditional = false, opts...) =
    supervised_window(register_window, θ, x, y, c; K, supervision, conditional, opts...)

end
