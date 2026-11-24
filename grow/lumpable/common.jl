# Exact entropies for a coarse-graining π of the characters. Everything in nats.
#
# A sample set is a list of (fine history id, next character, count). A history id is any
# integer naming the fine history; `hist[i]` holds its characters so that π can be applied.
#
#   class floor  F = H(πY | πH)
#   defect       D = I(πY ; H | πH) = H(πY | πH) − H(πY | H)      (0 ⇔ the coarse world is closed)
#   split        S = H(Y | πY, H)
#   chain rule   H(Y | H) = H(πY | H) + S = F − D + S

xlogx(c) = c > 0 ? c * log(c) : 0.0

"Σ c log c over the groups of `keys`, weights `w` (exact, by dictionary)."
function sumclogc(keys, w)
    d = Dict{eltype(keys),Float64}()
    for (k, c) in zip(keys, w)
        d[k] = get(d, k, 0.0) + c
    end
    sum(xlogx, values(d))
end

"H(A | B) from parallel vectors of keys and counts."
condent(a, b, w) = (sumclogc(b, w) - sumclogc(collect(zip(a, b)), w)) / sum(w)

"""
    quantities(hist, hid, y, w, cls) -> (F, D, S, fine, Hc_fine)

`hist[hid[i]]` is the fine history (a tuple or vector of characters) of sample i, `y[i]` the next
character, `w[i]` the count, `cls[a]` the class of character a.
"""
function quantities(hist, hid, y, w, cls)
    ch = [map(a -> cls[a], hist[h]) for h in hid]        # coarse histories
    cy = cls[y]
    F = condent(cy, ch, w)                                # H(πY | πH)
    G = condent(cy, hid, w)                               # H(πY | H)
    S = condent(y, collect(zip(cy, hid)), w)              # H(Y | πY, H)
    fine = condent(y, hid, w)                             # H(Y | H)
    (F = F, D = F - G, S = S, fine = fine, G = G)
end
