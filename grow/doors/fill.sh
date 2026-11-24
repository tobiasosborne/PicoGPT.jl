#!/bin/sh
# Print the tables quoted in RESULT.md from the result files.
cd "$(dirname "$0")"
echo "== steps to accuracy"; ./summary.sh
echo "== parents and gate gradient at birth (arms child-of-coarse, child-of-hops)"
LC_ALL=C awk -F'\t' 'FNR>1 && ($2=="child-of-coarse"||$2=="child-of-hops") {printf "seed %s %-16s parent steps %5d acc %.3f | birth max|dZ| %s | dl/dg FD %+.6f pullback %+.6f | zero-shot g=1 %.3f | final acc %.3f gate %+.3f\n",$1,$2,$3,$4,$5,$6,$7,$8,$11,$12}' results*.tsv | sort -k2n
