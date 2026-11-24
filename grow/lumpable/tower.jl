# Part 2 and 3: a tower of coarse-grainings of Tiny Shakespeare's 65 characters, read off from
# counts. Greedy, from 65 classes downwards: merge the two classes whose merger gives the smallest
# defect I(πY ; H | πH) (ties: smallest loss of I(πY ; πH)), H = the k previous characters.
#
#   julia -t2 grow/lumpable/tower.jl [path/to/tinyshakespeare.txt]
#
# Runs: k = 2 (trigram counts) and k = 1 (bigram counts) with the defect criterion, and for
# comparison Brown's criterion (smallest loss of class bigram information I(πY ; πH), k = 1).
# Writes tower_<run>.tsv, levels.tsv, tower.svg; prints the tables. Nats throughout.

include(joinpath(@__DIR__, "common.jl"))
using Printf

const ROOT = dirname(dirname(@__DIR__))
function corpus_path()
    isempty(ARGS) || return ARGS[1]
    for p in (joinpath(ROOT, "artifacts", "data", "tinyshakespeare.txt"),
              joinpath(homedir(), "Projects", "PicoGPT.jl", "artifacts", "data", "tinyshakespeare.txt"))
        isfile(p) && return p
    end
    error("Tiny Shakespeare not found; pass its path (scripts/common.jl downloads it to artifacts/data/)")
end
const text = read(corpus_path(), String)
const abc = sort(unique(text))
const V = length(abc)
const data = let d = Dict(c => a for (a, c) in enumerate(abc)); [d[c] for c in text]; end
const XL = [xlogx(Float64(c)) for c in 0:length(data)]       # table of c log c
@printf("corpus: %d characters, V = %d\n", length(data), V)

"Distinct (history, next) pairs with counts, for histories of the k previous characters."
function ngrams(k)
    cnt = Dict{NTuple{k + 1,Int},Int}()
    for i in k+1:length(data)
        g = ntuple(j -> data[i-k-1+j], k + 1)
        cnt[g] = get(cnt, g, 0) + 1
    end
    gs = sort(collect(keys(cnt)))
    hists = unique([g[1:k] for g in gs]); hnum = Dict(h => i for (i, h) in enumerate(hists))
    (k = k, hists = hists, hid = [hnum[g[1:k]] for g in gs], y = [g[end] for g in gs],
     h = [[g[j] for g in gs] for j in 1:k], c = [cnt[g] for g in gs], N = sum(values(cnt)))
end

# Fast evaluation: three sums of c log c over groupings of the n-grams, with dense scratch arrays.
struct Scratch; a::Vector{Int}; b::Vector{Int}; g::Vector{Int}; u::Vector{Int}; end
Scratch(G) = Scratch(zeros(Int, V^(G.k + 1)), zeros(Int, V^G.k), zeros(Int, length(G.hists) * V), zeros(Int, V))

"(Σ over (πH,πY), Σ over πH, Σ over (H,πY), Σ over πY) of c log c for the class vector cls."
function sums!(s::Scratch, G, cls)
    k = G.k; a = s.a; b = s.b; g = s.g; u = s.u
    @inbounds for i in eachindex(G.c)
        ch = k == 1 ? cls[G.h[1][i]] : (cls[G.h[1][i]] - 1) * V + cls[G.h[2][i]]
        cy = cls[G.y[i]]; c = G.c[i]
        a[(ch-1)*V+cy] += c; b[ch] += c; g[(G.hid[i]-1)*V+cy] += c; u[cy] += c
    end
    A = B = Gs = U = 0.0
    @inbounds for i in eachindex(G.c)
        ch = k == 1 ? cls[G.h[1][i]] : (cls[G.h[1][i]] - 1) * V + cls[G.h[2][i]]
        cy = cls[G.y[i]]
        ia = (ch - 1) * V + cy; ig = (G.hid[i] - 1) * V + cy
        A += XL[a[ia]+1]; a[ia] = 0
        B += XL[b[ch]+1]; b[ch] = 0
        Gs += XL[g[ig]+1]; g[ig] = 0
        U += XL[u[cy]+1]; u[cy] = 0
    end
    (A, B, Gs, U)
end

"Class floor F, defect D, split S, class information I(πY;πH), from the fast sums."
function fastq(s, G, cls, DH, E)
    A, B, Gs, U = sums!(s, G, cls)
    N = G.N
    F = (B - A) / N; Gc = (DH - Gs) / N
    (F = F, D = F - Gc, S = (Gs - E) / N, I = (xlogx(N) - U) / N - F)
end

function tower(G; criterion = :defect)
    DH = sumclogc(G.hid, G.c); E = sum(xlogx, Float64.(G.c))
    scr = [Scratch(G) for _ in 1:Threads.maxthreadid()]
    cls = collect(1:V)
    parts = Dict(V => copy(cls)); q = fastq(scr[1], G, cls, DH, E)
    rows = [(n = V, q..., merged = "")]
    for n in V:-1:2
        ids = sort(unique(cls))
        pairs = [(ids[i], ids[j]) for i in 1:n for j in i+1:n]
        res = Vector{Any}(undef, length(pairs))
        Threads.@threads :static for p in eachindex(pairs)
            pa, pb = pairs[p]          # (own names: `a, b` below would be shared across threads)
            c2 = copy(cls); c2[c2 .== pb] .= pa
            res[p] = fastq(scr[Threads.threadid()], G, c2, DH, E)
        end
        key(r) = criterion == :defect ? (round(r.D, digits = 12), -r.I) : (-r.I, r.D)
        best = argmin(p -> key(res[p]), eachindex(pairs))
        a, b = pairs[best]
        name(x) = escape_string(String(abc[cls .== x]))
        push!(rows, (n = n - 1, res[best]..., merged = name(a) * " + " * name(b)))
        cls[cls .== b] .= a
        parts[n-1] = copy(cls)
    end
    rows, parts
end

show_classes(cls) = join((escape_string(String(abc[cls .== x])) for x in sort(unique(cls), by = x -> -count(==(x), cls))), "  |  ")

"""
Level-relative quantities for a coarse partition c and a finer one f (f refines c):
  F_f = F_c − reldefect + relsplit, with
  reldefect = I(π_c Y ; π_f H | π_c H),  relsplit = H(π_f Y | π_c Y, π_f H),
and the split increment in the fine history, H(π_f Y | π_c Y, H).
"""
function relative(G, c, f)
    hf = [map(a -> f[a], h) for h in G.hists][G.hid]; hc = [map(a -> c[a], h) for h in G.hists][G.hid]
    yc = c[G.y]; yf = f[G.y]
    Fc = condent(yc, hc, G.c); Ff = condent(yf, hf, G.c)
    reldefect = Fc - condent(yc, hf, G.c)
    relsplit = condent(yf, collect(zip(yc, hf)), G.c)
    dS = condent(yf, collect(zip(yc, G.hid)), G.c)
    (Fc = Fc, Ff = Ff, reldefect = reldefect, relsplit = relsplit, dS = dS, err = abs(Ff - (Fc - reldefect + relsplit)))
end

const LEVELS = [32, 16, 8, 4, 2]
results = Dict{String,Any}()
levels_io = open(joinpath(@__DIR__, "levels.tsv"), "w")
println(levels_io, "run\tcoarse\tfine\tfloor_coarse\tfloor_fine\trel_defect\trel_split\tsplit_increment_fine_history\tbits_available_ln_ratio")
for (run, k, crit) in (("defect_k2", 2, :defect), ("defect_k1", 1, :defect), ("brown_k1", 1, :brown))
    G = ngrams(k)
    t = @elapsed ((rows, parts) = tower(G; criterion = crit))
    results[run] = rows
    @printf("\n================ run %s: history = %d previous character(s), criterion %s; %d distinct %d-grams, N = %d (%.1f s)\n",
            run, k, crit, length(G.c), k + 1, G.N, t)
    fine = rows[1].F
    @printf("fine floor H(Y | H) = %.6f nats\n", fine)
    open(joinpath(@__DIR__, "tower_$run.tsv"), "w") do io
        println(io, "classes\tclass_floor\tdefect\tsplit\tclass_information\tmerged")
        for r in rows; @printf(io, "%d\t%.9f\t%.9f\t%.9f\t%.9f\t%s\n", r.n, r.F, r.D, r.S, r.I, r.merged); end
    end
    # check the fast sums against the dictionary implementation, and the chain rule, at every level
    maxfast = maxchain = 0.0
    for r in rows
        qq = quantities(G.hists, G.hid, G.y, G.c, parts[r.n])
        maxfast = max(maxfast, abs(qq.F - r.F), abs(qq.D - r.D), abs(qq.S - r.S))
        maxchain = max(maxchain, abs(qq.fine - (qq.F - qq.D + qq.S)))
    end
    @printf("checks over all %d levels: fast vs dictionary max abs diff %.2e; chain rule F − D + S = fine floor max abs error %.2e\n",
            length(rows), maxfast, maxchain)
    println("\nclasses  class floor  defect     split      (defect / class floor)")
    for r in rows
        r.n in vcat(V, LEVELS, 1) && @printf("  %3d    %.6f     %.6f   %.6f   %.3f\n", r.n, r.F, r.D, r.S, r.F > 0 ? r.D / r.F : 0.0)
    end
    println("\nper level, coarse → fine (F_fine = F_coarse − rel.defect + rel.split; ΔS = split increment given the fine history; ln(fine/coarse) is only a reference scale):")
    println("  coarse → fine   F_coarse   F_fine     rel.defect  rel.split   ΔS         ln ratio   check")
    lv = vcat(1, reverse(LEVELS), V)
    for i in 1:length(lv)-1
        c, f = lv[i], lv[i+1]
        r = relative(G, parts[c], parts[f])
        @printf("  %3d → %3d      %.6f   %.6f   %.6f    %.6f    %.6f   %.4f     %.1e\n", c, f, r.Fc, r.Ff, r.reldefect, r.relsplit, r.dS, log(f / c), r.err)
        @printf(levels_io, "%s\t%d\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.6f\n", run, c, f, r.Fc, r.Ff, r.reldefect, r.relsplit, r.dS, log(f / c))
    end
    println("\nthe classes:")
    for n in LEVELS; println("  ", n, ":  ", show_classes(parts[n])); end
end
close(levels_io)

# ── one small plot: the three quantities against the number of classes, for the two defect runs ──
let W = 760, Hh = 420, L = 60, R = 20, Tm = 30, B = 45
    ymax = 2.6
    X(n) = L + (W - L - R) * log2(n) / log2(V); Y(v) = Hh - B - (Hh - B - Tm) * v / ymax
    io = IOBuffer()
    print(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $W $Hh" font-family="sans-serif" font-size="12">
<rect width="$W" height="$Hh" fill="white"/>\n""")
    for v in 0:0.5:2.5
        print(io, """<line x1="$L" x2="$(W-R)" y1="$(Y(v))" y2="$(Y(v))" stroke="#ddd"/><text x="$(L-6)" y="$(Y(v)+4)" text-anchor="end">$v</text>\n""")
    end
    for n in (1, 2, 4, 8, 16, 32, 65)
        print(io, """<line x1="$(X(n))" x2="$(X(n))" y1="$Tm" y2="$(Hh-B)" stroke="#eee"/><text x="$(X(n))" y="$(Hh-B+16)" text-anchor="middle">$n</text>\n""")
    end
    print(io, """<text x="$((W+L)/2)" y="$(Hh-8)" text-anchor="middle">number of classes (log scale)</text><text x="14" y="$(Hh/2)" transform="rotate(-90 14 $(Hh/2))" text-anchor="middle">nats per character</text>\n""")
    series = [("defect_k2", :F, "#1b6ca8", "", "class floor, k=2"), ("defect_k2", :D, "#c0392b", "", "defect, k=2"), ("defect_k2", :S, "#2e8b57", "", "split entropy, k=2"),
              ("defect_k1", :F, "#1b6ca8", "5 4", "class floor, k=1"), ("defect_k1", :D, "#c0392b", "5 4", "defect, k=1"), ("defect_k1", :S, "#2e8b57", "5 4", "split entropy, k=1")]
    for (j, (run, f, col, dash, label)) in enumerate(series)
        pts = join(("$(round(X(r.n), digits = 1)),$(round(Y(getfield(r, f)), digits = 1))" for r in results[run]), " ")
        print(io, """<polyline points="$pts" fill="none" stroke="$col" stroke-width="1.8" stroke-dasharray="$dash"/>\n""")
        print(io, """<line x1="$(L+15)" x2="$(L+45)" y1="$(Tm+2+16j)" y2="$(Tm+2+16j)" stroke="$col" stroke-width="1.8" stroke-dasharray="$dash"/><text x="$(L+52)" y="$(Tm+6+16j)">$label</text>\n""")
    end
    print(io, """<text x="$L" y="18" font-size="13">Tiny Shakespeare: greedy tower by least defect (history = k previous characters)</text>\n</svg>\n""")
    write(joinpath(@__DIR__, "tower.svg"), take!(io))
end
