"""
    Y, back = adjacent_block(X, p, H, T; trace)

A PicoGPT block with a two-token causal window. For every token, make a batch
entry [left neighbour, current token] and keep only the latter block output.
The first token sees a fixed zero column as its left neighbour. Scatter-add
the two pullback columns into their source positions, discarding the zero's.
"""
function adjacent_block(X, p, H, T; trace)
    D, N = size(X)
    pairs = zeros(eltype(X), D, 2N)
    pairs[:, 2:2:end] .= X
    for n in 2:N
        mod(n - 1, T) == 0 && continue
        @views pairs[:, 2n-1] .= X[:, n-1]
    end
    Y, back_pairs = block(pairs, p, H, 2; trace)
    if !isnothing(trace)
        local_A = only(trace).A
        A = zeros(eltype(X), T, T, H, N ÷ T)
        boundary_attention = zeros(eltype(X), H, N ÷ T)
        for b in 1:(N ÷ T), t in 1:T
            n = (b - 1) * T + t
            A[t, t, :, b] .= view(local_A, 2, 2, :, n)
            if t == 1
                boundary_attention[:, b] .= view(local_A, 2, 1, :, n)
            else
                A[t, t-1, :, b] .= view(local_A, 2, 1, :, n)
            end
        end
        trace[1] = (; A, boundary_attention)
    end
    function back(Ȳ)
        Ȳpairs = zero(Y)
        Ȳpairs[:, 2:2:end] .= Ȳ
        X̄pairs, p̄ = back_pairs(Ȳpairs)
        X̄ = X̄pairs[:, 2:2:end]
        for n in 2:N
            mod(n - 1, T) == 0 && continue
            @views X̄[:, n-1] .+= X̄pairs[:, 2n-1]
        end
        X̄, p̄
    end
    Y[:, 2:2:end], back
end
