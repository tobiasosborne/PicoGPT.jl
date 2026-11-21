"""
JLens — residual-depth Jacobians, character covectors and nonnegative J-space.

All derivatives are compositions of PicoGPT's hand-written pullbacks. Arrays use
one token per column; depth index j represents residual depth (j−1)/2.
"""
module JLens

using PicoGPT, LinearAlgebra, Random, Statistics

export residual_trace, attention_update, mlp_update, continue_from, pull_to_depth,
       jacobian_block, averaged_jacobians, lens_logits, lens_probs, lens_frame,
       nnls, exact_decomposition, nnomp, span_fraction, swap_coordinates,
       linear_cka, structure_statistics, read_json, specimen_data, depth_metadata,
       exact_space_statistics, greedy_space_statistics, specimen_interventions,
       lens_metrics, prompt_grid, prompt_swap, cosine_gram, unit_columns

"""Read the repository's JSON specimen without a non-stdlib dependency."""
function read_json(path)
    s = collect(read(path, String)); cursor = Ref(1)
    skip() = (while cursor[] ≤ length(s) && isspace(s[cursor[]]); cursor[] += 1; end)
    function value()
        skip(); ch = s[cursor[]]; cursor[] += 1
        if ch == '"'
            out = IOBuffer()
            while s[cursor[]] != '"'
                c = s[cursor[]]; cursor[] += 1
                if c == '\\'
                    c = s[cursor[]]; cursor[] += 1
                    if c == 'u'
                        c = Char(parse(UInt32, String(s[cursor[]:cursor[]+3]); base = 16))
                        cursor[] += 4
                    else
                        c = get(Dict('n'=>'\n', 'r'=>'\r', 't'=>'\t', 'b'=>'\b', 'f'=>'\f'), c, c)
                    end
                end
                print(out, c)
            end
            cursor[] += 1
            return String(take!(out))
        elseif ch == '[' || ch == '{'
            object = ch == '{'; result = object ? Dict{String,Any}() : Any[]
            skip(); closing = object ? '}' : ']'
            if s[cursor[]] == closing; cursor[] += 1; return result; end
            while true
                item = value()
                if object
                    skip(); @assert s[cursor[]] == ':'; cursor[] += 1
                    result[item] = value()
                else
                    push!(result, item)
                end
                skip(); c = s[cursor[]]; cursor[] += 1
                c == closing && return result
                @assert c == ','
            end
        else
            start = cursor[] - 1
            while cursor[] ≤ length(s) && !(s[cursor[]] in (',', ']', '}', ' ', '\n', '\r', '\t'))
                cursor[] += 1
            end
            word = String(s[start:cursor[]-1])
            word == "true" && return true
            word == "false" && return false
            word == "null" && return nothing
            return occursin(r"[.eE]", word) ? parse(Float64, word) : parse(Int, word)
        end
    end
    parsed = value(); skip(); @assert cursor[] == length(s) + 1
    parsed
end

"""Load exactly the trained parameters and batch in artifacts/specimen.json."""
function specimen_data(path)
    source = read_json(path); config = source["config"]
    c = Config(; (Symbol(k) => Int(config[k]) for k in ("V", "D", "H", "L", "T", "F"))...)
    x = hcat([Int.(v) for v in source["batch"]["x"]]...)
    y = hcat([Int.(v) for v in source["batch"]["y"]]...)
    (; c, θ = Float64.(source["train"]["theta_final"]), x, y,
       abc = only.(source["vocab"]), step = source["train"]["steps"])
end

"""
    Y, back = attention_update(X, p, H, T)

Y = X + Wo mix(softmax_rows(QᵀK/√d + M), V) + bo, with Q,K,V affine
maps of LN₁(X). Its adjoint adds the identity path to the branch pullback.
"""
function attention_update(X, p, H, T)
    Ahat, back_norm = layernorm(X, p.γ1, p.β1)
    Q, back_q = affine(Ahat, p.Wq, p.bq)
    K, back_k = affine(Ahat, p.Wk, p.bk)
    V, back_v = affine(Ahat, p.Wv, p.bv)
    A, back_attention = attention(Q, K, H, T)
    O, back_mix = mix(A, V)
    Y, back_out = affine(O, p.Wo, p.bo)
    function back(G)
        Gout = first(back_out(G))
        Ga, Gv = back_mix(Gout)
        Gq, Gk = back_attention(Ga)
        Gnorm = first(back_q(Gq)) + first(back_k(Gk)) + first(back_v(Gv))
        G + first(back_norm(Gnorm))
    end
    X + Y, back
end

"""
    Y, back = mlp_update(X, p)

Y = X + W2 relu(W1 LN₂(X) + b1) + b2. Each column is independent;
back(G) = G + D(MLP)(X)ᵀG, using the book's primitive pullbacks.
"""
function mlp_update(X, p)
    F, back_norm = layernorm(X, p.γ2, p.β2)
    U, back_in = affine(F, p.W1, p.b1)
    R, back_relu = relu(U)
    Y, back_out = affine(R, p.W2, p.b2)
    back(G) = G + first(back_norm(first(back_in(back_relu(first(back_out(G)))))))
    X + Y, back
end

"""Cache X₀, X½, …, X_L and one pullback per half block for repeated adjoints."""
function residual_trace(θ, tokens, c)
    p = carve(θ, c); X, _ = embed(tokens, p.E)
    states = [X]; backs = Any[]
    for b in p.blocks
        X, back = attention_update(X, b, c.H, size(tokens, 1))
        push!(states, X); push!(backs, back)
        X, back = mlp_update(X, b)
        push!(states, X); push!(backs, back)
    end
    (; states, backs, p, c, T = size(tokens, 1))
end

"""Pull G at X_L to depth index j: G_j = (∂X_L/∂X_j)ᵀG."""
function pull_to_depth(trace, G, j)
    for s in length(trace.backs):-1:j
        G = trace.backs[s](G)
    end
    G
end

"""Continue a replaced residual state at depth index j through the remaining maps."""
function continue_from(trace, X, j)
    for s in j:length(trace.backs)
        b = trace.p.blocks[(s+1) ÷ 2]
        X = first(isodd(s) ? attention_update(X, b, trace.c.H, trace.T) : mlp_update(X, b))
    end
    X
end

"""
    K = jacobian_block(trace, j, target, source)

K[i,q] = ∂X_L[i,target]/∂X_j[q,source]. Row i is the source column
of the pullback of e_i at target. The function also permits anti-causal tests.
"""
function jacobian_block(trace, j, target, source)
    D, N = size(trace.states[end]); K = zeros(D, D)
    for i in 1:D
        G = zeros(D, N); G[i, target] = 1
        K[i, :] = pull_to_depth(trace, G, j)[:, source]
    end
    K
end

"""
    result = averaged_jacobians(trace; self_targets = Int[])

J_j = (BT)⁻¹ Σ_{b,t} Σ_{t′≥t} ∂X_L[:,t′]/∂X_j[:,t].
This is (T+1)/2 times the literal uniform-pair average, so J_L = I.
A cotangent e_i at every output column sums all targets in one pullback.
Self estimates average K_tt over self_targets; future = J − self is the
mean over sources of the strictly later target sum, with the same denominator.
The split is exact when self_targets contains every token column.
"""
function averaged_jacobians(trace; self_targets = Int[])
    D, N = size(trace.states[end]); T = trace.T; B = N ÷ T
    S = length(trace.states); J = zeros(D, D, S)
    self = isempty(self_targets) ? nothing : zeros(D, D, S)
    for i in 1:D
        G = zeros(D, N); G[i, :] .= 1
        J[i, :, S] = vec(sum(G; dims = 2))
        for s in S-1:-1:1
            G = trace.backs[s](G)
            J[i, :, s] = vec(sum(G; dims = 2))
        end
        if self !== nothing
            for t in self_targets
                G = zeros(D, N); G[i, t] = 1
                self[i, :, S] .+= G[:, t]
                for s in S-1:-1:1
                    G = trace.backs[s](G)
                    self[i, :, s] .+= G[:, t]
                end
            end
        end
    end
    J ./= B * T
    future = nothing
    if self !== nothing
        self ./= length(self_targets)
        future = J - self
    end
    (; J, self, future)
end

depth_metadata(c) = (; depth = collect(0:0.5:c.L),
    depth_kind = ["embed"; repeat(["attention", "mlp"], c.L)])

"""Rows v_aᵀ of Wout J are character covectors; columns below are v_a."""
lens_frame(p, J) = Matrix((p.Wout * J)')

"""z(x) = Wout LN(Jx; γf,βf) + bout, with PicoGPT's fixed ε = 10⁻⁵."""
function lens_logits(p, X, J = nothing)
    Y = J === nothing ? X : J * X
    F, _ = layernorm(Y, p.γf, p.βf)
    first(affine(F, p.Wout, p.bout))
end
lens_probs(p, X, J = nothing) = softmax(lens_logits(p, X, J); dims = 1)
unit_columns(A) = A ./ max.(sqrt.(sum(abs2, A; dims = 1)), eps(Float64))
cosine_gram(A) = (U = unit_columns(A); U' * U)

"""
    c = nnls(A, x)

min_{c≥0} ‖x−Ac‖² by Lawson–Hanson's passive-set method. The KKT
conditions are c≥0, Aᵀ(x−Ac)≤0, c⊙Aᵀ(x−Ac)=0. Least squares is solved
on the passive columns; a line search returns offending entries to zero.
"""
function nnls(A, x; tolerance = 1e-11)
    n = size(A, 2); c = zeros(n); passive = falses(n)
    scale = max(norm(A) * norm(x), 1.0); tol = tolerance * scale
    for iteration in 1:10n+10
        w = A' * (x - A*c); w[passive] .= -Inf
        maximum(w; init = -Inf) ≤ tol && return c
        passive[argmax(w)] = true
        for inner in 1:10n+10
            z = zeros(n); z[passive] = A[:, passive] \ x
            if all(z[passive] .> 0)
                c = z; break
            end
            bad = findall(passive .& (z .≤ 0))
            α = minimum(c[i] / (c[i] - z[i]) for i in bad)
            c .+= α .* (z - c)
            inactive = passive .& (c .≤ tolerance)
            c[inactive] .= 0; passive[inactive] .= false
        end
    end
    error("NNLS failed to converge")
end

"""All k-subsets of 1:n, in lexicographic order (stdlib only)."""
function subsets(n, k)
    result = Vector{Vector{Int}}()
    function visit(prefix, start)
        if length(prefix) == k; push!(result, copy(prefix)); return; end
        for a in start:n-(k-length(prefix))+1
            push!(prefix, a); visit(prefix, a+1); pop!(prefix)
        end
    end
    visit(Int[], 1); result
end

"""
    subset, coefficients, component = exact_decomposition(A, x, k)

Enumerate every |S|=k and minimise ‖x−A_S c‖², c≥0. Ties retain the
first lexicographic subset; zero coefficients are allowed inside S.
"""
function exact_decomposition(A, x, k)
    best = Inf; chosen = Int[]; coefficients = Float64[]; component = zero(x)
    for S in subsets(size(A, 2), k)
        c = nnls(A[:, S], x); y = A[:, S] * c; distance = sum(abs2, x-y)
        if distance < best
            best = distance; chosen = S; coefficients = c; component = y
        end
    end
    chosen, coefficients, component
end

"""
    fractions, counts = nnomp(A, x)

Nonnegative orthogonal matching pursuit on unit columns: add the unselected
column maximising ⟨a,residual⟩>0, refit NNLS on the selected set, and repeat.
Return ‖x−x_J(k)‖²/‖x‖² and nonzero counts for k=0,…,V. A stalled
cone keeps its last residual for all remaining k.
"""
function nnomp(A, x)
    V = size(A, 2); selected = Int[]; used = falses(V)
    residual = copy(x); energy = sum(abs2, x)
    fractions = ones(V+1); counts = zeros(Int, V+1)
    for k in 1:V
        scores = A' * residual; scores[used] .= -Inf
        if maximum(scores) ≤ 1e-11 * max(norm(x), 1.0)
            fractions[k+1:end] .= fractions[k]; counts[k+1:end] .= counts[k]
            break
        end
        a = argmax(scores); push!(selected, a); used[a] = true
        c = nnls(A[:, selected], x)
        residual = x - A[:, selected]*c
        fractions[k+1] = sum(abs2, residual) / energy
        counts[k+1] = count(>(1e-10), c)
    end
    fractions, counts
end

"""Per-column energy ‖Π_span(A)x‖²/‖x‖², using a rank-revealing SVD."""
function span_fraction(A, X)
    F = svd(A); rank = count(>(maximum(F.S)*max(size(A)...)*eps(Float64)), F.S)
    Q = F.U[:, 1:rank]
    vec(sum(abs2, Q' * X; dims = 1) ./ sum(abs2, X; dims = 1))
end

"""
    edited, c, gram = swap_coordinates(x, V; α = 1)

c = V⁺x, x′ = x + αV(σ(c)−c). For independent columns,
V⁺ = (VᵀV)⁻¹Vᵀ; SVD least squares also handles a degenerate pair.
The component orthogonal to span(V) is unchanged.
"""
function swap_coordinates(x, V; α = 1.0)
    c = pinv(V) * x
    x + α .* (V * (reverse(c; dims = 1) - c)), c, V'V
end

"""
    linear_cka(X, Y)

CKA = ‖XcᵀYc‖²_F/(‖XcᵀXc‖_F ‖YcᵀYc‖_F), with columns
centred over the V character observations. Gram form avoids D×D products.
"""
function linear_cka(X, Y)
    Xc = X .- mean(X; dims = 1); Yc = Y .- mean(Y; dims = 1)
    K = Xc*Xc'; M = Yc*Yc'
    dot(K, M) / (norm(K)*norm(M))
end

"""Exact specimen cone fits and random-frame controls for every column and k."""
function exact_space_statistics(frames, states; seed = 431, draws = 20)
    rng = Xoshiro(seed); D, V = size(first(frames)); N = size(first(states), 2)
    random_frames = [unit_columns(randn(rng, D, V)) for _ in 1:draws]
    map(zip(frames, states)) do (A, X)
        selected = zeros(Int, V, V, N); coefficients = zeros(V, V, N)
        distance = zeros(V, N); norms = vec(sqrt.(sum(abs2, X; dims = 1)))
        for n in 1:N, k in 1:V
            S, c, component = exact_decomposition(A, X[:, n], k)
            selected[1:k, k, n] = S; coefficients[1:k, k, n] = c
            distance[k, n] = norm(X[:, n] - component)
        end
        random_fraction = zeros(V, N)
        for R in random_frames, n in 1:N, k in 1:V
            _, _, component = exact_decomposition(R, X[:, n], k)
            random_fraction[k, n] += sum(abs2, X[:, n] - component) / norms[n]^2 / draws
        end
        random_span = mean(reduce(hcat, [span_fraction(R, X) for R in random_frames]); dims = 2)[:]
        (; selected, coefficients, norm = norms, distance, random_fraction,
           span_fraction = span_fraction(A, X), random_span_fraction = random_span,
           frame_rank = rank(A), ambient_dimension = D, vocabulary_size = V)
    end
end

"""
    pursuit_curves(A, X)

The same NNOMP as nnomp, using G=AᵀA and B=AᵀX. A passive-set
NNLS refit starts from the preceding feasible coefficients. This avoids
repeating tall QR factorisations at every k and every validation position.
Only the small, positive-definite passive Gram system is solved anew.
"""
function pursuit_curves(A, X)
    G = A'A; correlations = A'X; V = size(A,2); N = size(X,2)
    fractions = ones(V+1,N); counts = zeros(Int,V+1,N)
    for n in 1:N
        b = correlations[:,n]; energy = sum(abs2,X[:,n]); c = zeros(V)
        selected = falses(V); passive = falses(V)
        tolerance = 1e-11 * max(sqrt(energy),1.0)
        for k in 1:V
            w = b-G*c; w[selected] .= -Inf
            if maximum(w) ≤ tolerance
                fractions[k+1:end,n] .= fractions[k,n]
                counts[k+1:end,n] .= counts[k,n]; break
            end
            candidate = argmax(w); selected[candidate] = true
            # Re-enter any previously inactive selected direction if its KKT
            # inequality becomes positive after the newly admitted direction.
            for outer in 1:10V
                w = b-G*c; w[.!selected .| passive] .= -Inf
                maximum(w) ≤ tolerance && break
                passive[argmax(w)] = true
                for inner in 1:10V
                    P = findall(passive); z = zeros(V)
                    system = cholesky(Symmetric(G[P,P]); check = false)
                    z[P] = issuccess(system) ? system \ b[P] : pinv(G[P,P])*b[P]
                    if all(z[P] .> 0); c = z; break; end
                    bad = [i for i in P if z[i] ≤ 0]
                    α = minimum(c[i]/(c[i]-z[i]) for i in bad)
                    c .+= α .* (z-c)
                    remove = passive .& (c .≤ 1e-12)
                    c[remove] .= 0; passive[remove] .= false
                end
            end
            fractions[k+1,n] = max(0.0,(energy-2dot(c,b)+dot(c,G*c))/energy)
            counts[k+1,n] = count(>(1e-10),c)
        end
    end
    fractions, counts
end

"""Mean NNOMP curves, occupancy and span energies over validation activations."""
function greedy_space_statistics(frames, states; seed = 937, draws = 5)
    D, V = size(first(frames)); rng = Xoshiro(seed)
    random_frames = [unit_columns(randn(rng, D, V)) for _ in 1:draws]
    result = Vector{Any}(undef, length(frames))
    # Depths are independent; keep BLAS single-threaded inside this outer parallelism.
    Threads.@threads for j in eachindex(frames)
        A = unit_columns(frames[j]); X = states[j]; N = size(X, 2)
        fractions, counts = pursuit_curves(A, X)
        nonzero25 = counts[26,:]
        random_curve = zeros(V+1)
        for R in random_frames
            curves, _ = pursuit_curves(R, X)
            random_curve .+= vec(mean(curves; dims = 2)) ./ draws
        end
        curve = vec(mean(fractions; dims = 2))
        improvement = -diff(curve); random_improvement = -diff(random_curve)
        occupancy = findfirst(improvement .< random_improvement)
        result[j] = (; k = collect(0:V), residual_fraction = curve,
            random_residual_fraction = random_curve, improvement, random_improvement,
            occupancy, median_nonzero_at_25 = median(nonzero25), positions = N,
            span_fraction = mean(span_fraction(A, X)),
            random_span_fraction = mean(mean(span_fraction(R, X)) for R in random_frames))
        println("NNOMP depth ", (j-1)/2, " complete"); flush(stdout)
    end
    result
end

"""Validation readout scores; excess kurtosis is across characters, then averaged."""
function lens_metrics(p, X, J, model_logits, targets)
    Z = lens_logits(p, X, J); P = softmax(Z; dims = 1)
    M = softmax(model_logits; dims = 1); N = length(targets)
    top = [argmax(z) for z in eachcol(Z)]
    model_top = [argmax(z) for z in eachcol(model_logits)]
    logP = Z .- [logsumexp(z) for z in eachcol(Z)]'
    logM = model_logits .- [logsumexp(z) for z in eachcol(model_logits)]'
    centred = Z .- mean(Z; dims = 1)
    kurtosis = vec(mean(centred.^4; dims = 1) ./ mean(centred.^2; dims = 1).^2 .- 3)
    (; positions = N, model_agreement = mean(top .== model_top),
       top1_accuracy = mean(top .== vec(targets)),
       top3_accuracy = mean(targets[n] in partialsortperm(Z[:, n], 1:3; rev = true) for n in 1:N),
       kl_model_lens = sum(M .* (logM - logP))/N,
       mean_excess_kurtosis = mean(kurtosis))
end

"""Top-three character ids and probabilities, with axes [rank,position,depth]."""
function prompt_grid(trace, J)
    V = trace.c.V; N = size(trace.states[1], 2); S = length(trace.states)
    ids = zeros(Int, 3, N, S, 2); probabilities = zeros(3, N, S, 2)
    for j in 1:S, lens in 1:2
        P = lens_probs(trace.p, trace.states[j], lens == 1 ? J[:, :, j] : nothing)
        for n in 1:N
            selected = partialsortperm(P[:, n], 1:3; rev = true)
            ids[:, n, j, lens] = selected; probabilities[:, n, j, lens] = P[selected, n]
        end
    end
    model = lens_probs(trace.p, trace.states[end])
    (; ids, probabilities, model_top1 = [argmax(p) for p in eachcol(model)],
       model_final = model[:, end])
end

"""Choose a depth-two pair by exhaustive α=1 swaps; retain the largest final TV change."""
function prompt_swap(trace, frame; j = 5)
    X = trace.states[j]; baseline = lens_probs(trace.p, trace.states[end])[:, end]
    original_top = argmax(baseline); best_changed = false; best_distance = -Inf
    best_pair = (1, 2)
    for a in 1:trace.c.V-1, b in a+1:trace.c.V
        edited, _, _ = swap_coordinates(X, frame[:, [a,b]])
        P = lens_probs(trace.p, continue_from(trace, edited, j))[:, end]
        changed = argmax(P) != original_top; distance = sum(abs, P-baseline)/2
        if (changed && !best_changed) || (changed == best_changed && distance > best_distance)
            best_changed = changed; best_distance = distance; best_pair = (a,b)
        end
    end
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0]
    probabilities = hcat([begin
        edited, _, _ = swap_coordinates(X, frame[:, collect(best_pair)]; α)
        lens_probs(trace.p, continue_from(trace, edited, j))[:, end]
    end for α in alphas]...)
    (; depth = (j-1)/2, pair = collect(best_pair), alphas, probabilities,
       baseline, changed_top1_at_one = best_changed, total_variation_at_one = best_distance,
       selection = "Enumerate every unordered pair at alpha=1; prefer a changed top-1, then maximise final total variation.")
end

"""Quartiles and median of a numerical population (normalised beforehand)."""
summary_stats(values) = (; q25 = quantile(values, 0.25), median = median(values), q75 = quantile(values, 0.75))

"""Mean reciprocal cosine rank, with tied labels assigned their average rank."""
function label_preservation(OV, A)
    C = unit_columns(OV*A)' * unit_columns(A)
    mean(1 / (1 + count(>(C[i,i] + 1e-12), C[i,:]) +
               (count(x -> abs(x-C[i,i]) ≤ 1e-12, C[i,:])-1)/2) for i in axes(C, 1))
end

"""
    structure_statistics(p, c, frames)

MLP gain is ‖W2 relu(W1 LN₂(v)+b1)+b2‖ divided by the random median.
OV_h = Wo[:,h] Wv[h,:]. Its gain uses the random mean; label preservation
subtracts the random reciprocal-rank baseline. All directions have unit norm.
"""
function structure_statistics(p, c, frames; seed = 1229, random_count = 2000)
    rng = Xoshiro(seed); R = unit_columns(randn(rng, c.D, random_count))
    rotation = Matrix(qr(randn(rng, c.D, c.D)).Q)
    random_labels = R[:, 1:c.V]; mlp = Any[]; ov = Any[]
    for b in 1:c.L
        block = p.blocks[b]
        populations = Dict("j_lens" => unit_columns(frames[2b]),
            "logit_lens" => unit_columns(Matrix(p.Wout')), "random" => R)
        b > 1 && (populations["previous_neurons"] = unit_columns(Matrix(p.blocks[b-1].W2)))
        branch(A) = first(mlp_update(A, block)) - A
        baseline = median(vec(sqrt.(sum(abs2, branch(R); dims = 1))))
        push!(mlp, (; block = b, random_median = baseline,
            populations = Dict(name => summary_stats(vec(sqrt.(sum(abs2, branch(A); dims = 1))) ./ baseline)
                               for (name,A) in populations)))
        A = unit_columns(frames[2b-1])
        populations_ov = Dict("j_lens" => A, "rotated_j_lens" => rotation*A,
            "logit_lens" => unit_columns(Matrix(p.Wout')), "random" => random_labels)
        for h in 1:c.H
            rows = (h-1)*(c.D÷c.H)+1:h*(c.D÷c.H)
            OV = block.Wo[:, rows] * block.Wv[rows, :]
            random_gain = mean(vec(sqrt.(sum(abs2, OV*R; dims = 1))))
            random_label = label_preservation(OV, random_labels)
            statistics = Dict(name => (; gain = mean(vec(sqrt.(sum(abs2, OV*P; dims = 1))))/random_gain,
                label_preservation = label_preservation(OV, P)-random_label,
                raw_reciprocal_rank = label_preservation(OV, P)) for (name,P) in populations_ov)
            push!(ov, (; block = b, head = h, random_gain, random_label_baseline = random_label, populations = statistics))
        end
    end
    (; seed, random_count, random_label_count = c.V, mlp, attention_ov = ov)
end

"""
    specimen_interventions(traces, frames, targets)

Steering, projection ablation, two-covector swaps, and equal-norm concept writes.
A shared table interns probability columns at six significant digits. Readout ids
are indexed by [column,character,alpha] for steering at the edited token,
[offset,position,character,alpha] for later steering in window 1,
[offset,column,character] for ablation and [offset,column,pair] for swaps.
Offset index 1 means the edited token; zero ids pad beyond a window. Later
steering is sampled to keep the complete export below the size budget.
"""
function specimen_interventions(traces, frames, targets)
    T = traces[1].T; D, V = size(first(frames)); N = length(targets)
    table = Vector{Vector{Float64}}(); lookup = Dict{NTuple{V,Float64},Int}()
    function intern(p)
        key = Tuple(round.(p; sigdigits = 6))
        get!(lookup, key) do
            push!(table, collect(key)); length(table)
        end
    end
    baseline = [lens_probs(tr.p, tr.states[end]) for tr in traces]
    pairs = [[a,b] for a in 1:V-1 for b in a+1:V]; alphas = collect(-4:4)
    result = Any[]
    for j in eachindex(frames)
        A = frames[j]; unit = unit_columns(A)
        Xall = hcat([tr.states[j] for tr in traces]...)
        steer = zeros(Int, N, V, length(alphas))
        steer_later_window1 = zeros(Int, T, T, V, length(alphas))
        ablate = zeros(Int, T, N, V)
        swap = zeros(Int, T, N, length(pairs)); coordinates = zeros(2, N, length(pairs))
        grams = zeros(2, 2, length(pairs))
        function readout!(destination, replacement, n)
            b = (n-1) ÷ T + 1; t = (n-1) % T + 1; tr = traces[b]
            X = copy(tr.states[j]); X[:,t] = replacement
            P = lens_probs(tr.p, continue_from(tr, X, j))
            for target in t:min(T,t+length(destination)-1)
                destination[target-t+1] = intern(P[:, target])
            end
        end
        for n in 1:N, a in 1:V
            x = Xall[:,n]
            for (i, α) in enumerate(alphas)
                if n ≤ T
                    readout!(view(steer_later_window1,:,n,a,i), x + α*A[:,a], n)
                    steer[n,a,i] = steer_later_window1[1,n,a,i]
                else
                    readout!(view(steer,n:n,a,i), x + α*A[:,a], n)
                end
            end
            readout!(view(ablate,:,n,a), x-dot(x,unit[:,a])*unit[:,a], n)
        end
        for (i, pair) in enumerate(pairs), n in 1:N
            replacement, c, gram = swap_coordinates(Xall[:,n], A[:,pair])
            coordinates[:,n,i] = c; grams[:,:,i] = gram
            readout!(view(swap,:,n,i), replacement, n)
        end
        concepts = zeros(D, V, 3); changes = zeros(V, 3); concept_norm = zeros(V)
        for a in 1:V
            concept = vec(mean(Xall[:, vec(targets).==a]; dims = 2) - mean(Xall; dims = 2))
            _, _, component = exact_decomposition(A, concept, 3)
            concept_norm[a] = norm(concept)
            directions = hcat(component, concept-component, A[:,a])
            concepts[:,a,:] = directions
            eligible = findall(vec(targets) .!= a)
            for kind in 1:3
                direction = directions[:,kind]
                write = norm(direction) > 1e-14 ? direction * (norm(concept)/norm(direction)) : zero(direction)
                for n in eligible
                    b = (n-1) ÷ T + 1; t = (n-1) % T + 1; tr = traces[b]
                    X = copy(tr.states[j]); X[:,t] .+= write
                    Z = lens_logits(tr.p, continue_from(tr,X,j))[:,t]
                    Z0 = lens_logits(tr.p, tr.states[end])[:,t]
                    changes[a,kind] += ((Z[a]-logsumexp(Z))-(Z0[a]-logsumexp(Z0))) / length(eligible)
                end
            end
        end
        push!(result, (; steer, steer_later_window1, ablate, swap, swap_coordinates = coordinates, swap_gram = grams,
            concept_vectors = concepts, concept_norm, concept_mean_delta_logp = changes))
        println("Specimen interventions depth ", (j-1)/2, " complete"); flush(stdout)
    end
    (; alphas, pairs, probability_table = hcat(table...), by_depth = result,
       concept_components = ["j_k3", "remainder", "lens_vector"])
end

end # module
