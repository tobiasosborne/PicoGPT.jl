"""A unit vector in 1⟂, orthogonal to all preceding columns of a Helmert basis."""
function state_axis(D, j)
    v = zeros(D)
    v[1:j] .= 1
    v[j+1] = -j
    v / norm(v)
end

"""
Construct a local XOR transition on a finite set of stream states. The block's
uniform two-key attention writes the sum of the two state signs into a separate
axis. A scalar projection separates the resulting states, and a ReLU spline
interpolates the residual update that maps each state to the desired next state.
The spline has flat intervals around the interpolation sites.
"""
function structured_init(c, task; perturbation = 0.0, seed = 1)
    task in (:prefix, :rule60) || throw(ArgumentError("unknown local task"))
    c.V == 2 && c.H == 1 || throw(ArgumentError("the local construction uses two symbols and one head"))
    prefix_task = task == :prefix
    required = prefix_task ? (6, 22) : (4, 10)
    c.L == 1 && c.D ≥ required[1] && c.F ≥ required[2] ||
        throw(ArgumentError("the structured initialization needs D ≥ $(required[1]), F ≥ $(required[2])"))
    θ = zeros(nparams(c))
    p = carve(θ, c)
    b = only(p.blocks)
    b.γ1 .= b.γ2 .= p.γf .= 1
    b.Wo .= I(c.D)
    scale = sqrt(c.D * (1 - 1e-5) / (prefix_task ? 3 : 2))
    marker = scale * state_axis(c.D, 1)
    state = scale * state_axis(c.D, 2)
    value = sqrt(c.D) * state_axis(c.D, 3)
    input = prefix_task ? scale * state_axis(c.D, 4) : zero(state)
    for (a, sign) in enumerate((1, -1))
        p.E[:, a] .= marker + sign * (state + input)
        p.Wout[a, :] .= sign * state / dot(state, state)
    end
    b.Wv .= value * state' / dot(state, state)

    examples = [(; raw, current, left) for raw in (prefix_task ? (1, -1) : (0,)),
                current in (1, -1), left in (1, 0, -1)]
    mids = stack(vec(map(examples) do (; raw, current, left)
        marker + raw * input + current * state + ((current + left) / 2) * value
    end))
    targets = stack(vec(map(examples) do (; raw, current, left)
        previous = left == 0 ? 1 : left
        if prefix_task
            # The injected embedding supplies marker, input, and the raw sign.
            (raw * previous - raw) * state
        else
            marker + current * previous * state
        end
    end))
    normalized, _ = layernorm(mids, b.γ2, b.β2)
    updates = targets - mids
    rng = Xoshiro(909)
    # Choose the projection with the largest minimum gap among fixed random draws.
    candidates = [normalize(randn(rng, c.D)) for _ in 1:256]
    direction = argmax(w -> minimum(diff(sort(vec(w' * normalized)))), candidates)
    scores = vec(direction' * normalized)
    order = sortperm(scores)
    b.b2 .= updates[:, first(order)]
    for (j, (a, z)) in enumerate(zip(order[1:end-1], order[2:end]))
        left, right = scores[a], scores[z]
        lo, hi = 0.8left + 0.2right, 0.2left + 0.8right
        slope = (updates[:, z] - updates[:, a]) / (hi - lo)
        b.W1[2j-1, :] .= b.W1[2j, :] .= direction
        b.b1[2j-1], b.b1[2j] = -lo, -hi
        b.W2[:, 2j-1] .= slope
        b.W2[:, 2j] .= -slope
    end
    θ .+ perturbation .* randn(Xoshiro(seed), length(θ))
end

const LOCAL_TRANSITION = (; attention_span = :adjacent, position_code = false, stabilize = true)
const LOCAL_RECIPE = (; RECIPE..., steps = 120, batch = 16,
                         η = 0.001, ηmin = 0.0001, warmup = 10, λ = 0.0)
const LOCAL_PERTURBATION = 0.002

const CAUTIOUS_RECIPE = (; LOCAL_RECIPE..., steps = 60, η = 1e-5, ηmin = 1e-6, warmup = 5)
const CAUTIOUS_PERTURBATION = 5e-5
