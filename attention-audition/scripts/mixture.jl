include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../../jspace/src/JLens.jl"))
using .JLens: read_json, mlp_update, lens_logits
using Statistics: mean, median

const AUDITION = dirname(@__DIR__)
const TOL = 1e-12

function readout(X0, O, parameters)
    b = only(parameters.blocks)
    attention_output, _ = affine(O, b.Wo, b.bo)
    Xout, _ = mlp_update(X0 + attention_output, b)
    lens_logits(parameters, Xout)
end

max_error(x, y) = maximum(abs, x .- y)
kl(p, q) = max(0.0, sum(p .* (log.(p) .- log.(q))))
tv(p, q) = sum(abs, p .- q) / 2
stats(x) = (; mean = mean(x), median = median(x), max = maximum(x))

function analyze_reader(b, i, x, y, vocab, forward, parameters, affine_readout)
    (; X0, V, A, O, Z, P) = forward
    T = size(x, 1)
    n = (b - 1) * T + i
    d = size(O, 1) ÷ 2
    head1, head2 = 1:d, d+1:2d
    columns = (b - 1) * T .+ (1:i)
    a = A[i, :, :, b]
    w1, w2 = a[1:i, 1], a[1:i, 2]
    v1, v2 = V[head1, columns], V[head2, columns]
    o1, o2 = O[head1, n], O[head2, n]
    p = P[:, n]
    target = y[i, b]
    Xreader = X0[:, n:n]
    other = [j for j in axes(O, 2) if j != n]
    edited = copy(O)
    local_logit_error = 0.0
    other_logit_error = 0.0
    other_probability_error = 0.0
    changed_logits = 0.0

    # Evaluate only the reader column, then repeat over the full batch to check locality.
    function intervene(source_output)
        z = readout(Xreader, reshape(source_output, :, 1), parameters)[:, 1]
        edited[:, n] = source_output
        full_z = readout(X0, edited, parameters)
        full_p = softmax(full_z; dims = 1)
        local_logit_error = max(local_logit_error, max_error(z, full_z[:, n]))
        other_logit_error = max(other_logit_error, max_error(full_z[:, other], Z[:, other]))
        other_probability_error = max(other_probability_error, max_error(full_p[:, other], P[:, other]))
        changed_logits = max(changed_logits, max_error(z, Z[:, n]))
        softmax(z)
    end

    pair_weights = w1 * w2'
    p_by_pair = Array{Float64}(undef, length(p), i, i)
    affine_by_pair = similar(p_by_pair)
    for j2 in 1:i, j1 in 1:i
        output = [v1[:, j1]; v2[:, j2]]
        p_by_pair[:, j1, j2] = intervene(output)
        affine_by_pair[:, j1, j2] = affine_readout(output)
    end
    pairs = reshape(p_by_pair, length(p), :)
    p_mix = pairs * vec(pair_weights)
    p_geo = softmax(log.(pairs) * vec(pair_weights))

    p_by_source = Array{Float64}(undef, length(p), i, 2)
    for j in 1:i
        p_by_source[:, j, 1] = intervene([v1[:, j]; o2])
        p_by_source[:, j, 2] = intervene([o1; v2[:, j]])
    end
    p_single = hcat(p_by_source[:, :, 1] * w1, p_by_source[:, :, 2] * w2)

    affine_geo = softmax(reshape(log.(affine_by_pair), length(p), :) * vec(pair_weights))
    affine_error = max_error(affine_readout(O[:, n]), affine_geo)

    # Verify the one-source case before recording the algebraically identical probabilities.
    one_source_error = 0.0
    if i == 1
        @assert w1 == w2 == [1.0]
        one_source_error = maximum(q -> max_error(p, q),
                                   (p_mix, p_geo, eachcol(p_single)..., p_by_pair[:, 1, 1]))
        @assert one_source_error ≤ TOL
        p_mix = p_geo = p
        p_single .= p
        p_by_pair[:, 1, 1] = p
        p_by_source[:, 1, :] .= p
    end

    entropy(w) = -sum(z -> iszero(z) ? 0.0 : z * log(z), w)
    spread(w, v, o) = sum(w[j] * sum(abs2, v[:, j] - o) for j in eachindex(w))
    diagnostics = (; entropy = [entropy(w1), entropy(w2)],
                    weighted_value_spread = [spread(w1, v1, o1), spread(w2, v2, o2)],
                    max_weight = [maximum(w1), maximum(w2)])
    distances = (; kl_p_mix = kl(p, p_mix), kl_mix_p = kl(p_mix, p), kl_p_geo = kl(p, p_geo),
                 tv_p_mix = tv(p, p_mix), tv_p_geo = tv(p, p_geo), tv_mix_geo = tv(p_mix, p_geo))
    argmax_ids = (; p = argmax(p), mix = argmax(p_mix), geo = argmax(p_geo),
                  single = map(argmax, eachcol(p_single)))
    argmax_agreement = (; p_mix = argmax_ids.p == argmax_ids.mix,
                        p_geo = argmax_ids.p == argmax_ids.geo,
                        mix_geo = argmax_ids.mix == argmax_ids.geo,
                        all = argmax_ids.p == argmax_ids.mix == argmax_ids.geo)
    cross_entropy = (; p = -log(p[target]), mix = -log(p_mix[target]), geo = -log(p_geo[target]))
    single_head = map(eachcol(p_single)) do q
        (; kl_p_mix = kl(p, q), kl_mix_p = kl(q, p), tv_p_mix = tv(p, q),
           argmax_agreement = argmax(p) == argmax(q), cross_entropy = -log(q[target]))
    end
    checks = (; affine_error, local_logit_error, other_logit_error, other_probability_error,
              changed_logits, one_source_error)
    (; b, i, window = decode(x[:, b], vocab), target_id = target, target = string(vocab[target]),
       attention_rows = a, pair_weights, p, p_mix, p_geo, p_single, p_by_source, p_by_pair,
       distances, argmax_ids, argmax_agreement, cross_entropy, diagnostics, single_head, checks)
end

function summarize(readers)
    distance_keys = keys(first(readers).distances)
    agreement_keys = keys(first(readers).argmax_agreement)
    agreement(n) = (; count = n, total = length(readers), fraction = n / length(readers))
    distances = (; (key => stats([r.distances[key] for r in readers]) for key in distance_keys)...)
    argmax_agreement = (; (key => agreement(count(r -> r.argmax_agreement[key], readers))
                          for key in agreement_keys)...)
    mean_cross_entropy = (; (key => mean(r -> r.cross_entropy[key], readers)
                            for key in keys(first(readers).cross_entropy))...)
    diagnostics = [(;
        (key => stats([r.diagnostics[key][h] for r in readers])
         for key in keys(first(readers).diagnostics))...) for h in 1:2]
    single_head = [(
        distances = (; (key => stats([r.single_head[h][key] for r in readers])
                        for key in (:kl_p_mix, :kl_mix_p, :tv_p_mix))...),
        argmax_agreement = agreement(count(r -> r.single_head[h].argmax_agreement, readers)),
        mean_cross_entropy = mean(r -> r.single_head[h].cross_entropy, readers),
    ) for h in 1:2]
    worst = (; (key => [(; r.b, r.i, r.window, value = r.distances[key])
                       for r in sort(readers; by = r -> r.distances[key], rev = true)[1:5]]
                for key in distance_keys)...)
    (; distances, argmax_agreement, mean_cross_entropy, single_head, diagnostics, worst)
end

function figure_readers(readers)
    # Qualify a head by two individually substantial sources with distinct characters.
    spreads = NamedTuple[]
    for h in 1:2
        candidates = filter(readers) do r
            chars = collect(r.window)
            a = r.attention_rows[:, h]
            strong = findall(≥(0.15), a)
            length(unique(chars[strong])) ≥ 2 && maximum(a) ≤ 0.75 &&
                all(s -> (s.b, s.i) != (r.b, r.i), spreads)
        end
        isempty(candidates) && continue
        r = argmax(r -> r.single_head[h].tv_p_mix, candidates)
        push!(spreads, (; kind = "spread", head = h, r...))
    end
    @assert !isempty(spreads)
    near = argmax(r -> minimum(r.diagnostics.max_weight), filter(r -> r.i > 1, readers))
    @assert minimum(near.diagnostics.max_weight) ≥ 0.99
    [spreads; (; kind = "near-copy", head = nothing, near...)]
end

pack(xs::AbstractArray{<:Union{Number,AbstractString}}) = tensor(xs)
pack(xs::AbstractArray) = pack.(xs)
pack(x::NamedTuple) = (; (k => pack(v) for (k, v) in pairs(x))...)
pack(x) = x

function main()
    BLAS.set_num_threads(1)
    source = read_json(artifact("specimen.json"))
    config_keys = ("V", "D", "H", "L", "T", "F")
    c = Config(; (Symbol(k) => Int(source["config"][k]) for k in config_keys)...)
    x, y = (hcat([Int.(row) for row in source["batch"][key]]...) for key in ("x", "y"))
    θ = Float64.(source["train"]["theta_final"])
    vocab = only.(source["vocab"])
    @assert c.L == 1 && c.H == 2 && size(x) == size(y) == (c.T, 10) && length(x) == 80

    parameters = carve(θ, c)
    block_parameters = only(parameters.blocks)
    # Fixed affine readout obtained by composing the two trained projections and biases.
    M = parameters.Wout * block_parameters.Wo
    bias = parameters.Wout * block_parameters.bo + parameters.bout
    affine_readout(o) = softmax(M * o + bias)

    trace = []
    Z, _ = gpt(θ, x, c; trace)
    (; V, A, O) = trace[2]
    (; X0) = trace[1]
    P = softmax(Z; dims = 1)
    forward = (; X0, V, A, O, Z, P)
    baseline_error = max_error(readout(X0, O, parameters), Z)
    @assert baseline_error ≤ TOL

    readers = [analyze_reader(b, i, x, y, vocab, forward, parameters, affine_readout)
               for b in axes(x, 2) for i in axes(x, 1)]
    summary = summarize(readers)
    figures = figure_readers(readers)
    stored_loss = last(source["train"]["loss"])
    package_loss = first(loss(θ, x, y, c))
    loss_error = maximum(abs, [stored_loss - package_loss,
                               summary.mean_cross_entropy.p - package_loss])
    maxima = (; (key => maximum(r -> r.checks[key], readers) for key in keys(first(readers).checks))...)
    @assert all(maxima[key] ≤ TOL for key in (:affine_error, :local_logit_error,
                :other_logit_error, :other_probability_error, :one_source_error))
    @assert maxima.changed_logits > TOL && loss_error ≤ TOL
    checks = (; stored_loss, package_loss, loss_error, baseline_error, maxima...,
              interventions = sum(r.i^2 + 2 * r.i for r in readers), one_source_readers = size(x, 2))

    conventions = (;
        tensors = "Column-major {shape,data}. Vocabulary ids, windows b and positions i are 1-based.",
        readers = "reader index i+(b-1)T; window text includes all T input characters",
        attention_rows = "[source, head], with zeros after reader i",
        pair_weights = "[source in head 1, source in head 2], both sources in 1:i",
        p_by_pair = "[vocab, source in head 1, source in head 2]",
        p_by_source = "[vocab, source, head replaced], with the other head held at its usual output",
        p_single = "[vocab, head sampled], the weighted arithmetic mean of p_by_source",
        diagnostics = "[head]; entropy in nats, weighted_value_spread is an average squared Euclidean distance",
        distances = "KL and cross-entropy use natural logarithms, TV is half the L1 distance",
    )
    result = (; source = "artifacts/specimen.json train.theta_final", train_steps = source["train"]["steps"],
              config = (; (Symbol(k) => source["config"][k] for k in config_keys)...),
              vocab = string.(vocab), windows = [decode(x[:, b], vocab) for b in axes(x, 2)],
              reader_count = length(readers), conventions, summary, checks, readers, figure_readers = figures)
    writejson(joinpath(AUDITION, "artifacts", "mixture.json"), pack(result))
    write_notes(joinpath(AUDITION, "notes", "numbers.md"), result)
end

function write_notes(path, result)
    (; summary, checks, reader_count, figure_readers, vocab) = result
    labels = (; kl_p_mix = "KL(p‖mix)", kl_mix_p = "KL(mix‖p)", kl_p_geo = "KL(p‖geo)",
              tv_p_mix = "TV(p, mix)", tv_p_geo = "TV(p, geo)", tv_mix_geo = "TV(mix, geo)")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# The mean-field step in the trained specimen\n")
        println(io, "Equal weight for each of the $reader_count readers in the ten overlapping training windows. Each variant uses the saved trained parameters and the same attention rows, values and residual input.")
        print(io, raw"""

        At reader $i$, independently draw $j_1,j_2\le i$ with weight $w=A^{(1)}_{ij_1}A^{(2)}_{ij_2}$. Let $q$ be the next-character distribution after substituting those two value vectors into column $i$. The usual prediction $p$ decodes the two mean value vectors. The arithmetic mixture $p^{\rm mix}=\sum wq$ averages the per-pair probabilities. The geometric mixture $p^{\rm geo}=\operatorname{softmax}(\sum w\log q)$ averages their logarithms and renormalises. A single-head mixture draws one source and keeps the other head at its usual mean output.

        ## Distribution gaps

        KL and target cross-entropy are in nats. TV is half the sum of absolute probability differences.

        | Distance | Mean | Median | Max |
        | --- | ---: | ---: | ---: |
        """)
        for (key, label) in pairs(labels)
            s = summary.distances[key]
            @printf(io, "| %s | %.6g | %.6g | %.6g |\n", label, s.mean, s.median, s.max)
        end
        println(io, "\n| Prediction | Mean target cross-entropy |\n| --- | ---: |")
        for (key, label) in pairs((; p = "Usual", mix = "Arithmetic mixture", geo = "Geometric mixture"))
            @printf(io, "| %s | %.9f |\n", label, summary.mean_cross_entropy[key])
        end
        agree(key) = "$(summary.argmax_agreement[key].count)/$reader_count"
        println(io, "\nTop character agreement was $(agree(:p_mix)) for p and mix, $(agree(:p_geo)) for p and geo, and $(agree(:mix_geo)) for mix and geo. All three agreed at $(agree(:all)) readers.")
        if summary.argmax_agreement.all.count != reader_count
            for r in result.readers
                r.argmax_agreement.all && continue
                (; p, mix, geo) = r.argmax_ids
                println(io, "At window $(r.b) `$(repr(r.window))`, position $(r.i), the three top characters were `$(repr(vocab[p]))`, `$(repr(vocab[mix]))` and `$(repr(vocab[geo]))`, respectively.")
            end
        end
        println(io, "\n| Head sampled | Mean KL(p‖mix) | Mean KL(mix‖p) | Mean TV | Top character agreement with p | Mean target cross-entropy |\n| ---: | ---: | ---: | ---: | ---: | ---: |")
        for (h, s) in enumerate(summary.single_head)
            @printf(io, "| %d | %.6g | %.6g | %.6g | %d/%d | %.6g |\n", h,
                    s.distances.kl_p_mix.mean, s.distances.kl_mix_p.mean, s.distances.tv_p_mix.mean,
                    s.argmax_agreement.count, reader_count, s.mean_cross_entropy)
        end
        println(io, "\n| Head | Mean row entropy (nats) | Mean weighted squared value spread | Mean maximum weight |\n| ---: | ---: | ---: | ---: |")
        for (h, s) in enumerate(summary.diagnostics)
            @printf(io, "| %d | %.6g | %.6g | %.6g |\n", h, s.entropy.mean,
                    s.weighted_value_spread.mean, s.max_weight.mean)
        end

        println(io, "\n## Largest gaps\n\nThe three highest values for each distance, with the full window and its 1-based reader position.\n")
        println(io, "| Distance | Largest | Second | Third |\n| --- | --- | --- | --- |")
        for (key, label) in pairs(labels)
            cells = [@sprintf("`%s` · %d · %.6g", repr(r.window), r.i, r.value)
                     for r in summary.worst[key][1:3]]
            println(io, "| $label | $(join(cells, " | ")) |")
        end

        println(io, "\n## Figure readers\n")
        println(io, "Each spread example maximises the selected head's single-head TV among readers with at least 0.15 weight on each of two distinct characters and no individual source above 0.75. The examples are selected in head order with distinct readers. The near-copy example maximises the smaller of the two row maxima among readers with more than one allowed source.\n")
        for r in figure_readers
            heading = "Window $(r.b) `$(repr(r.window))`, position $(r.i)"
            chars = collect(r.window)
            if r.kind == "spread"
                h = r.head
                sources = [@sprintf("%d (`%s`, %.4f)", j, repr(chars[j]), r.attention_rows[j, h])
                           for j in 1:r.i if r.attention_rows[j, h] ≥ 0.15]
                @printf(io, "- **%s.** Head %d gives substantial weight to %s. Sampling that head gives TV %.6g from p.\n",
                        heading, h, join(sources, ", "), r.single_head[h].tv_p_mix)
            else
                j1, j2 = (argmax(r.attention_rows[:, h]) for h in 1:2)
                @printf(io, "- **%s.** The two heads put %.6f and %.6f on their favourite sources at positions %d (`%s`) and %d (`%s`). TV between p and the joint arithmetic mixture is %.6g.\n",
                        heading, r.diagnostics.max_weight..., j1, repr(chars[j1]), j2, repr(chars[j2]), r.distances.tv_p_mix)
            end
        end
        for r in figure_readers
            all(==(r.argmax_ids.p), r.argmax_ids.single) && continue
            a = r.target_id
            p, q, g = r.p[a], r.p_mix[a], r.p_geo[a]
            @printf(io, "\nAt window %d `%s`, position %d, the probability of the target `%s` is %.4f under p, %.4f under the arithmetic mixture and %.4f under the geometric mixture. The single-head mixtures choose `%s` and `%s`, respectively.\n",
                    r.b, repr(r.window), r.i, repr(vocab[a]), p, q, g, repr.(vocab[r.argmax_ids.single])...)
        end
        println(io, "\nThe figure records in [mixture.json](../artifacts/mixture.json) carry both attention rows, per-source predictions for each head with the other held fixed, and all three joint predictions. Numeric arrays use column-major storage.")

        println(io, "\n## Checks\n")
        @printf(io, "- The ordinary mean loss agrees with the stored final checkpoint loss (%.12f nats) and the package loss to within %.3g nats. The brief gives approximately 0.0588 nats.\n",
                checks.stored_loss, checks.loss_error)
        @printf(io, "- Replaying the unedited batch from the attention output reproduced the package logits to %.3g. Across all %d interventions, untouched columns changed by at most %.3g in logits and %.3g in probabilities. The edited-column logits agreed with the isolated-column calculation to %.3g.\n",
                checks.baseline_error, checks.interventions, checks.other_logit_error,
                checks.other_probability_error, checks.local_logit_error)
        @printf(io, "- With fixed affine logits, averaging log probabilities subtracts a common scalar from the expected logits. Softmax cancels that scalar, so decoding the mean and taking the normalised geometric mixture agree. On the specimen's actual rows and values, the largest probability error was %.3g, below 1e-12. For the real nonlinear readout, mean TV between p and geo was %.6g and the maximum was %.6g.\n",
                checks.affine_error, summary.distances.tv_p_geo.mean, summary.distances.tv_p_geo.max)
        @printf(io, "- At position 1, both heads have a single source in all %d windows. The variants coincide algebraically. The raw calculation agreed within %.3g before recording the identical distributions.\n",
                checks.one_source_readers, checks.one_source_error)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
