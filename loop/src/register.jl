register_nparams(c::Config) = c.L * (4c.F + 1)

"""One two-input, one-output ReLU MLP per block, initialized with Gaussian weights."""
function register_init(c::Config, R = Float64; rng)
    θ = zeros(R, register_nparams(c))
    for p in register_blocks(θ, c)
        randn!(rng, p.W1)
        randn!(rng, p.W2)
        p.W1 ./= sqrt(R(2))
        p.W2 ./= sqrt(R(c.F))
    end
    θ
end

function register_blocks(θ, c)
    stride = 4c.F + 1
    length(θ) == stride * c.L || throw(DimensionMismatch("wrong register parameter count"))
    F = c.F
    [let offset = (l-1)*stride
         (; W1 = reshape(view(θ, offset+1:offset+2F), F, 2),
            b1 = view(θ, offset+2F+1:offset+3F),
            W2 = reshape(view(θ, offset+3F+1:offset+4F), 1, F),
            b2 = view(θ, offset+4F+1:offset+stride))
     end for l in 1:c.L]
end

"""
    Z, back = registergpt(θ, x, c; K, tied, input, bounded, readouts, trace)

Replace the block with hard lag-one attention over a scalar bit register and a
pointwise ReLU MLP. The MLP sees the attended scalar and either the immutable input
(:source) or the current scalar (:state). Its output is the binary logit difference,
clipped to [-1,1] for feedback when bounded=true. Input bits use -1/+1, the initial
state is the input, and the predecessor of position one is the zero bit (-1).
"""
function registergpt(θ, x, c::Config; K::Int = c.L, tied::Bool = true, input = :source,
                     bounded::Bool = true, readouts = :last, trace = nothing)
    c.V == 2 && c.D == 2 && c.H == 1 || throw(ArgumentError("use V=D=2 and H=1"))
    K ≥ 1 && (tied ? c.L == 1 : c.L ≥ K) ||
        throw(ArgumentError("use L=1 for tied passes or L≥K for untied passes, with K ≥ 1"))
    readouts in (:last, :all) || throw(ArgumentError("unknown readout mode"))
    input in (:source, :state) || throw(ArgumentError("unknown register input"))
    T, B = size(x)
    N = length(x)
    R = eltype(θ)
    source = reshape(R.(2 .* x .- 3), 1, :)
    state = source
    record!(trace, (; pass = 0, X = state))
    Zs = typeof(source)[]
    back_passes = Vector{Function}(undef, K)
    blocks = register_blocks(θ, c)

    A = if !isnothing(trace)
        weights = zeros(R, T, T, 1, B)
        for t in 2:T
            weights[t, t-1, :, :] .= 1
        end
        weights
    end
    for k in 1:K
        p = blocks[tied ? 1 : k]
        pair = fill(-one(R), 2, N)
        pair[1, :] .= vec(input == :source ? source : state)
        for b in 1:B
            offset = (b-1)*T
            @views pair[2, offset+2:offset+T] .= state[1, offset+1:offset+T-1]
        end
        U, back_U = affine(pair, p.W1, p.b1)
        H, back_H = relu(U)
        u, back_u = affine(H, p.W2, p.b2)
        state = bounded ? clamp.(u, -one(R), one(R)) : u
        back_passes[k] = function (ū_direct, statē)
            ū = bounded ? ū_direct + statē .* (abs.(u) .< 1) : ū_direct + statē
            H̄, W̄2, b̄2 = back_u(ū)
            pair̄, W̄1, b̄1 = back_U(back_H(H̄))
            incoming = input == :source ? zeros(R, 1, N) : pair̄[1:1, :]
            for b in 1:B
                offset = (b-1)*T
                @views incoming[1, offset+1:offset+T-1] .+= pair̄[2, offset+2:offset+T]
            end
            incoming, (W̄1, b̄1, W̄2, b̄2)
        end
        if readouts == :all || k == K || !isnothing(trace)
            Z = vcat(zero(u), u)
            (readouts == :all || k == K) && push!(Zs, Z)
            if !isnothing(trace)
                record!(trace, (; pass = k, X = state, A, Z, predictions = predictions(Z, T),
                                  boundary_attention = ones(R, 1, B)))
            end
        end
    end

    function back(Z̄)
        θ̄ = zero(θ)
        gradients = register_blocks(θ̄, c)
        statē = zero(source)
        for k in K:-1:1
            ū = readouts == :all ? Z̄[k][2:2, :] : k == K ? Z̄[2:2, :] : zero(source)
            statē, p̄ = back_passes[k](ū, statē)
            for (dst, src) in zip(values(gradients[tied ? 1 : k]), p̄)
                dst .+= src
            end
        end
        θ̄
    end
    (readouts == :all ? Zs : only(Zs)), back
end

registerloss(θ, x, y, c; supervision = :last, opts...) =
    supervised_loss(readouts -> registergpt(θ, x, c; readouts, opts...), x, y, supervision)
