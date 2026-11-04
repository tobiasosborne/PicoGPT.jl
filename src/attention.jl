# Chapter 5 · Attention
#
# Everything else in the model acts on each column of X ∈ ℝ^{D×N} separately. Attention
# is the one place where tokens meet: column i of its output is an average of the columns
# j ≤ i of the same sequence. So it is also the one place that must know which columns
# form a sequence, and which rows form a head:
#
#   heads      the D rows split into H blocks of d = D/H, a direct sum ℝᴰ = ℝᵈ ⊕ ⋯ ⊕ ℝᵈ,
#              and each block (head) attends on its own;
#   sequences  the N = T·B columns split into B blocks of T.
#
# That leaves H·B independent problems, each on d×T slices.

"""
    foreach_head(f, A, Xs...)

For every head h and sequence b, call f on the (h, b) slices of A and of the arrays Xs:
the T×T matrix X[:, :, h, b] of a T×T×H×B array, or the d×T block (rows of head h,
columns of sequence b) of a D×N array. The H·B calls are independent, so they run in
parallel; meanwhile BLAS is kept single-threaded, as our threads already fill the cores.
"""
function foreach_head(f, A, Xs...)
    T, _, H, B = size(A)
    slice(X::AbstractArray{<:Any,4}, h, b) = view(X, :, :, h, b)
    slice(X::AbstractMatrix, h, b) = (d = size(X, 1) ÷ H; view(X, (h-1)*d+1:h*d, (b-1)*T+1:b*T))
    without_blas_threads() do
        Threads.@threads for (h, b) in collect(Iterators.product(1:H, 1:B))
            f(map(X -> slice(X, h, b), (A, Xs...))...)
        end
    end
end

# BLAS's thread count is global. Overlapping and nested scopes share one setting;
# only the last scope to leave restores it. The lock protects entry/exit, not the work.
const BLAS_SCOPE = (lock = ReentrantLock(), users = Ref(0), threads = Ref(0))
function without_blas_threads(f)
    lock(BLAS_SCOPE.lock) do
        if BLAS_SCOPE.users[] == 0
            BLAS_SCOPE.threads[] = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
        end
        BLAS_SCOPE.users[] += 1
    end
    try
        f()
    finally
        lock(BLAS_SCOPE.lock) do
            BLAS_SCOPE.users[] -= 1
            BLAS_SCOPE.users[] == 0 && BLAS.set_num_threads(BLAS_SCOPE.threads[])
        end
    end
end

"""
    A, back = attention(Q, K, H, T)

The causal attention matrices of every head and sequence, A ∈ ℝ^{T×T×H×B}. For head h of
sequence b, with queries q and keys k the d×T slices of Q and K,

    S = qᵀk / √d,     A = softmax_rows(S + M),     Mᵢⱼ = −∞ if j > i, else 0.

Sᵢⱼ = qᵢ⋅kⱼ/√d scores how well key j answers query i. Row i of A is a probability
distribution over the keys j ≤ i: a token may read from its past, never from its future,
since exp(−∞) = 0 puts A in the lower triangle.

Pullback (Ā ↦ Q̄, K̄). Row by row, softmax has dAᵢⱼ = Aᵢⱼ (dSᵢⱼ − Σₖ Aᵢₖ dSᵢₖ), so
S̄ᵢⱼ = Aᵢⱼ (Āᵢⱼ − Σₖ Aᵢₖ Āᵢₖ), which vanishes where Aᵢⱼ = 0: no gradient from the future
either. Then dS = (dqᵀk + qᵀdk)/√d gives Q̄ = k S̄ᵀ/√d and K̄ = q S̄/√d.
"""
function attention(Q, K, H, T)
    D, N = size(Q)
    A = similar(Q, T, T, H, N ÷ T)
    α = 1 / √eltype(Q)(D ÷ H)                             # 1/√d, in the number type of Q
    M = eltype(Q)[j > i ? -Inf : 0 for i in 1:T, j in 1:T]     # the causal mask
    foreach_head(A, Q, K) do a, q, k
        mul!(a, q', k, α, false)                          # S = qᵀk/√d
        softmax!(a .+= M; dims = 2)                       # A = softmax_rows(S + M)
    end
    function back(Ā)
        S̄, Q̄, K̄ = similar(A), similar(Q), similar(K)
        foreach_head(A, Ā, S̄, Q, K, Q̄, K̄) do a, ā, s̄, q, k, q̄, k̄
            s̄ .= a .* ā                                    # S̄ = A ⊙ Ā …
            s̄ .-= a .* sum(s̄, dims = 2)                   # … − A ⊙ (row sums of A ⊙ Ā)
            mul!(q̄, k, s̄', α, false)                     # Q̄ = k S̄ᵀ/√d
            mul!(k̄, q, s̄, α, false)                      # K̄ = q S̄ /√d
        end
        Q̄, K̄
    end
    A, back
end

"""
    O, back = mix(A, V)

Each head mixes the value columns of its sequence by its attention matrix,

    O = V Aᵀ,    oᵢ = Σⱼ Aᵢⱼ vⱼ,

so output column i is a convex combination of the value columns j ≤ i. Stacking the
heads' outputs (the direct sum again) gives O ∈ ℝ^{D×N}.

Pullback (Ō ↦ Ā, V̄). ⟨Ō, dV Aᵀ + V dAᵀ⟩ = ⟨Ō A, dV⟩ + ⟨Ōᵀ V, dA⟩.
"""
function mix(A, V)
    O = similar(V)
    foreach_head(A, V, O) do a, v, o
        mul!(o, v, a')                                    # O = V Aᵀ
    end
    function back(Ō)
        Ā, V̄ = similar(A), similar(V)
        foreach_head(A, V, Ō, Ā, V̄) do a, v, ō, ā, v̄
            mul!(ā, ō', v)                                # Ā = Ōᵀ V
            mul!(v̄, ō, a)                                 # V̄ = Ō A
        end
        Ā, V̄
    end
    O, back
end
