"""Causal dot-product attention with a fixed relative logit at lag one."""
function biased_attention(Q, K, H, T, previous_bias)
    D, N = size(Q)
    A = similar(Q, T, T, H, N ÷ T)
    α = inv(sqrt(eltype(Q)(D ÷ H)))
    M = eltype(Q)[j > i ? -Inf : j == i - 1 ? previous_bias : 0 for i in 1:T, j in 1:T]
    PicoGPT.foreach_head(A, Q, K) do a, q, k
        mul!(a, q', k, α, false)
        PicoGPT.softmax!(a .+= M; dims = 2)
    end
    function back(Ā)
        Q̄, K̄ = similar(Q), similar(K)
        PicoGPT.foreach_head(A, Ā, Q, K, Q̄, K̄) do a, ā, q, k, q̄, k̄
            S̄ = a .* (ā .- sum(a .* ā; dims = 2))
            mul!(q̄, k, S̄', α, false)
            mul!(k̄, q, S̄, α, false)
        end
        Q̄, K̄
    end
    A, back
end

"""
The stock block with an alternate attention route. In the local route one
extra zero column supplies the boundary to every sequence. The same vector
projection serves the left and current slots.
"""
function relative_block(X, p, H, T, span, bias; trace)
    D, N = size(X)
    local_window = span != :full
    boundary = N + 1
    left = [mod1(n, T) == 1 ? boundary : n - 1 for n in 1:N]
    U = local_window ? hcat(X, zeros(eltype(X), D)) : X
    F, back_F = layernorm(U, p.γ1, p.β1)
    V, back_V = affine(F, p.Wv, p.bv)
    if span == :previous
        O = V[:, left]
        function hard_back(Ō)
            V̄ = zero(V)
            for n in 1:N
                @views V̄[:, left[n]] .+= Ō[:, n]
            end
            zero(p.Wq), zero(p.bq), zero(p.Wk), zero(p.bk), zero(F), V̄
        end
        back_attention = hard_back
        if !isnothing(trace)
            A = zeros(eltype(X), T, T, H, N ÷ T)
            for t in 2:T
                A[t, t-1, :, :] .= 1
            end
            record!(trace, (; A, boundary_attention = ones(eltype(X), H, N ÷ T)))
        end
    else
        Q, back_Q = affine(F, p.Wq, p.bq)
        K, back_K = affine(F, p.Wk, p.bk)
        if local_window
            O, back_mix = two_key_mix(Q, K, V, H, T, left, bias; trace)
        else
            A, back_A = biased_attention(Q, K, H, T, bias)
            O, mix_pull = mix(A, V)
            back_mix = function (Ō)
                Ā, V̄ = mix_pull(Ō)
                Q̄, K̄ = back_A(Ā)
                Q̄, K̄, V̄
            end
            record!(trace, (; A))
        end
        back_attention = function (Ō)
            Q̄, K̄, V̄ = back_mix(Ō)
            F̄q, W̄q, b̄q = back_Q(Q̄)
            F̄k, W̄k, b̄k = back_K(K̄)
            W̄q, b̄q, W̄k, b̄k, F̄q + F̄k, V̄
        end
    end
    Y₁, back_Y₁ = affine(O, p.Wo, p.bo)
    X₁ = X + Y₁
    G, back_G = layernorm(X₁, p.γ2, p.β2)
    M, back_M = affine(G, p.W1, p.b1)
    R, back_R = relu(M)
    Y₂, back_Y₂ = affine(R, p.W2, p.b2)
    function back(X̄₂)
        R̄, W̄2, b̄2 = back_Y₂(X̄₂)
        Ḡ, W̄1, b̄1 = back_M(back_R(R̄))
        X̄₁, γ̄2, β̄2 = back_G(Ḡ)
        X̄₁ .+= X̄₂
        Ō, W̄o, b̄o = back_Y₁(X̄₁)
        W̄q, b̄q, W̄k, b̄k, F̄qk, V̄ = back_attention(Ō)
        F̄v, W̄v, b̄v = back_V(V̄)
        Ū, γ̄1, β̄1 = back_F(F̄qk + F̄v)
        X̄ = X̄₁ + view(Ū, :, 1:N)
        X̄, (γ̄1, β̄1, W̄q, b̄q, W̄k, b̄k, W̄v, b̄v, W̄o, b̄o, γ̄2, β̄2, W̄1, b̄1, W̄2, b̄2)
    end
    X₁ + Y₂, back
end

"""Two-key softmax written as a sigmoid of the left-minus-self score."""
function two_key_mix(Q, K, V, H, T, left, bias; trace)
    D, N = size(Q)
    N -= 1
    d = D ÷ H
    α = inv(sqrt(eltype(Q)(d)))
    A = similar(Q, H, N)
    O = similar(Q, D, N)
    for n in 1:N, h in 1:H
        rows = (h-1)*d+1:h*d
        l = left[n]
        Δ = sum(i -> Q[i,n] * (K[i,l] - K[i,n]), rows) * α + bias
        a = inv(1 + exp(-Δ))
        A[h,n] = a
        for i in rows
            O[i,n] = a * V[i,l] + (1-a) * V[i,n]
        end
    end
    if !isnothing(trace)
        dense = zeros(eltype(Q), T, T, H, N ÷ T)
        boundary_attention = similar(Q, H, N ÷ T)
        for n in 1:N
            t, b = mod1(n, T), cld(n, T)
            dense[t,t,:,b] .= 1 .- view(A, :, n)
            if t == 1
                boundary_attention[:,b] .= view(A, :, n)
            else
                dense[t,t-1,:,b] .= view(A, :, n)
            end
        end
        record!(trace, (; A = dense, boundary_attention))
    end
    function back(Ō)
        Q̄, K̄, V̄ = zero(Q), zero(K), zero(V)
        for n in 1:N, h in 1:H
            rows = (h-1)*d+1:h*d
            l, a = left[n], A[h,n]
            Δ̄ = α * a * (1-a) * sum(i -> Ō[i,n] * (V[i,l] - V[i,n]), rows)
            for i in rows
                Q̄[i,n] = Δ̄ * (K[i,l] - K[i,n])
                K̄[i,l] += Δ̄ * Q[i,n]
                K̄[i,n] -= Δ̄ * Q[i,n]
                V̄[i,l] += a * Ō[i,n]
                V̄[i,n] += (1-a) * Ō[i,n]
            end
        end
        Q̄, K̄, V̄
    end
    O, back
end
