# 1 · The smallest exact case: the loss floors of the fine and the coarse text.
#
#     julia --project=. grow/tower/floors.jl
#
# The specimen's world is the 80 (context, next symbol) pairs of the ten windows of length
# 8 of "to be or not to be". A floor is a conditional entropy of that empirical
# distribution: what the best possible predictor, given what it sees, still pays.

include("common.jl")

data = encode(CORPUS, ABC)
T = 8
x, y = windows(data, T)
N = length(x)

# every prediction: its fine history h, its coarse history πh, the target a and its class v
cases = [(h = x[1:t, b], πh = CLS[x[1:t, b]], a = y[t, b], v = CLS[y[t, b]])
         for b in axes(x, 2) for t in 1:T]

"""
The conditional entropy H(target | seen), as N·H = Σ_contexts Σ_outcomes k ln(n/k) with
integer counts: returned as exact integer coefficients of ln 2, ln 3, ln 5, ln 7.
"""
function floor_exact(seen, target)
    groups = Dict{Any,Vector{Any}}()
    for case in cases
        push!(get!(groups, seen(case), []), target(case))
    end
    primes = (2, 3, 5, 7)
    coeff = zeros(Int, length(primes))
    function addlog!(m, weight)                    # coeff += weight · ln m, by factorising m
        for (i, p) in enumerate(primes)
            while m % p == 0
                coeff[i] += weight
                m ÷= p
            end
        end
        m == 1 || error("a count has a prime factor above 7")
    end
    ambiguous = 0
    for outcomes in values(groups)
        n = length(outcomes)
        for a in unique(outcomes)
            k = count(==(a), outcomes)
            addlog!(n, k)
            addlog!(k, -k)
        end
        length(unique(outcomes)) > 1 && (ambiguous += n)
    end
    (; coeff, value = sum(coeff .* log.(primes)) / N, contexts = length(groups), ambiguous)
end

function describe(f)
    terms = ["$(k) ln $(p)" for (k, p) in zip(f.coeff, (2, 3, 5, 7)) if k != 0]
    expr = isempty(terms) ? "0" : "(" * replace(join(terms, " + "), "+ -" => "− ") * ")/$N"
    @sprintf("%-28s = %.12f nats   (%d contexts, %d of %d predictions undetermined)",
             expr, f.value, f.contexts, f.ambiguous, N)
end

floors = (
    fine          = floor_exact(c -> c.h,         c -> c.a),   # H(a | h): the specimen's floor
    coarse_coarse = floor_exact(c -> c.πh,        c -> c.v),   # H(v | πh): the coarse text, seen coarsely
    coarse_fine   = floor_exact(c -> c.h,         c -> c.v),   # H(v | h): the class, given the fine history
    refine_fine   = floor_exact(c -> (c.v, c.h),  c -> c.a),   # H(a | v, h): the refinement term
    refine_coarse = floor_exact(c -> (c.v, c.πh), c -> c.a),   # H(a | v, πh): a refinement blind to the fine past
    fine_coarse   = floor_exact(c -> c.πh,        c -> c.a),   # H(a | πh): the fine text read coarsely
)

if abspath(PROGRAM_FILE) == @__FILE__
    println("fine text    ", repr(CORPUS))
    println("coarse text  ", repr(join("VCS"[CLS[data]])), "   (V vowel, C consonant, S space)\n")
    println("H(a | h)      fine floor                     ", describe(floors.fine))
    println("H(v | πh)     coarse floor, coarse history   ", describe(floors.coarse_coarse))
    println("H(v | h)      coarse floor, fine history     ", describe(floors.coarse_fine))
    println("H(a | v, h)   refinement, fine history       ", describe(floors.refine_fine))
    println("H(a | v, πh)  refinement, coarse history     ", describe(floors.refine_coarse))
    println("H(a | πh)     fine text, coarse history      ", describe(floors.fine_coarse))

    specimen = (2log(2) + 3log(3)) / 80
    chain = floors.coarse_fine.value + floors.refine_fine.value
    println()
    @printf("fine floor − (2 ln 2 + 3 ln 3)/80           = %+.1e\n", floors.fine.value - specimen)
    @printf("chain rule: H(v|h) + H(a|v,h) − H(a|h)       = %+.1e   (integer coefficients equal: %s)\n",
            chain - floors.fine.value, floors.coarse_fine.coeff + floors.refine_fine.coeff == floors.fine.coeff)
    @printf("chain rule: H(v|πh) + H(a|v,πh) − H(a|πh)    = %+.1e   (integer coefficients equal: %s)\n",
            floors.coarse_coarse.value + floors.refine_coarse.value - floors.fine_coarse.value,
            floors.coarse_coarse.coeff + floors.refine_coarse.coeff == floors.fine_coarse.coeff)
    gap = floors.coarse_coarse.value - floors.coarse_fine.value
    @printf("\nwhat the coarse history hides: H(v|πh) − H(v|h) = I(v; h | πh) = %.12f nats\n", gap)
    @printf("floor of a frozen coarse parent with a fine refinement: H(v|πh) + H(a|v,h) = %.12f nats\n",
            floors.coarse_coarse.value + floors.refine_fine.value)
    @printf("  that is %.2f × the fine floor %.12f\n",
            (floors.coarse_coarse.value + floors.refine_fine.value) / floors.fine.value, floors.fine.value)
end
