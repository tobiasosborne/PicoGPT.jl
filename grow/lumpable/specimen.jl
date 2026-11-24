# Part 1: every partition of the specimen's seven characters into 2, 3 and 4 classes.
#   julia grow/lumpable/specimen.jl        (writes specimen.tsv, prints the summary)
#
# Samples as in the repository (src/train.jl `windows`, scripts/export_specimen.jl): the text has
# 18 characters, a window of length T = 8 starts at each of i = 1…10, and position t of window i
# predicts character i+t from the prefix text[i : i+t-1]. That is N = 80 (history, next) pairs
# with histories of length 1…8, and the fine floor is (2 ln 2 + 3 ln 3)/80.

include(joinpath(@__DIR__, "common.jl"))
using Printf

const corpus = "to be or not to be"
const T = 8
const abc = sort(unique(corpus))
const V = length(abc)
const data = [findfirst(==(c), abc) for c in corpus]

hists = Vector{Vector{Int}}(); hidx = Dict{Vector{Int},Int}()
hid = Int[]; y = Int[]; w = Float64[]
for i in 1:length(data)-T, t in 1:T
    h = data[i:i+t-1]
    id = get!(hidx, h) do; push!(hists, h); length(hists); end
    push!(hid, id); push!(y, data[i+t]); push!(w, 1.0)
end
N = length(y)
floor_exact = (2log(2) + 3log(3)) / 80

"All partitions of 1:n as class vectors in restricted-growth form."
function partitions(n)
    out = Vector{Vector{Int}}()
    rec(c, m) = length(c) == n ? push!(out, copy(c)) :
        for k in 1:m+1; push!(c, k); rec(c, max(m, k)); pop!(c); end
    rec(Int[], 0); out
end
show_classes(cls) = join((String([abc[a] == ' ' ? '␣' : abc[a] for a in 1:V if cls[a] == k]) for k in 1:maximum(cls)), "|")

rows = []
maxerr = 0.0
for cls in partitions(V)
    k = maximum(cls)
    q = quantities(hists, hid, y, w, cls)
    global maxerr = max(maxerr, abs(q.fine - (q.F - q.D + q.S)), abs(q.fine - floor_exact))
    q.D < -1e-12 && error("negative defect")
    push!(rows, (k = k, name = show_classes(cls), cls = cls, q...))
end
@printf("N = %d samples, V = %d, %d partitions in all\n", N, V, length(rows))
@printf("fine floor %.15f, exact (2ln2+3ln3)/80 = %.15f\n", rows[1].fine, floor_exact)
@printf("chain rule F − D + S = fine floor: max abs error over all partitions %.2e\n", maxerr)

open(joinpath(@__DIR__, "specimen.tsv"), "w") do io
    println(io, "classes\tpartition\tclass_floor\tdefect\tsplit\tclass_floor_given_fine")
    for r in sort(rows, by = r -> (r.k, r.D, r.F))
        @printf(io, "%d\t%s\t%.12f\t%.12f\t%.12f\t%.12f\n", r.k, r.name, r.F, r.D, r.S, r.G)
    end
end

vcs = [abc[a] == ' ' ? 1 : abc[a] in "aeiou" ? 3 : 2 for a in 1:V]   # restricted-growth order: ␣ | bnrt | eo
vcs_name = show_classes(vcs)
for k in 2:4
    R = sort(filter(r -> r.k == k, rows), by = r -> (r.D, r.F))
    closed = count(r -> r.D < 1e-12, R)
    @printf("\n%d classes: %d partitions, %d closed (defect < 1e-12); median defect %.5f, max %.5f\n",
            k, length(R), closed, R[(end+1)÷2].D, R[end].D)
    println("   rank  defect     class floor  split      partition")
    for (i, r) in enumerate(R)
        (i <= 8 || i == length(R) || r.name == vcs_name) &&
            @printf("   %4d  %.6f   %.6f     %.6f   %s%s\n", i, r.D, r.F, r.S, r.name,
                    r.name == vcs_name ? "   <- vowel|consonant|space" : "")
    end
    if k == 3
        i = findfirst(r -> r.name == vcs_name, R)
        @printf("   vowel|consonant|space: rank %d of %d by defect; %d partitions have strictly smaller defect\n",
                i, length(R), count(r -> r.D < R[i].D - 1e-12, R))
    end
end
println("\nsmallest defect at each number of classes (fine floor $(round(floor_exact, digits = 6))):")
for k in 1:V
    R = sort(filter(r -> r.k == k, rows), by = r -> (r.D, r.F))
    @printf("   k=%d  %3d partitions  min defect %.6f  (class floor %.6f, split %.6f)  %s\n",
            k, length(R), R[1].D, R[1].F, R[1].S, R[1].name)
end
