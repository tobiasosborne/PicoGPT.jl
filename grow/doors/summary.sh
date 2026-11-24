#!/bin/sh
# First evaluation step (every 25) at which test accuracy on the fine world reaches each level, per seed and arm.
cd "$(dirname "$0")"
LC_ALL=C awk -F'\t' 'FNR>1 { k=$1" "$2; if(!(k in seen)){seen[k]=1; order[++n]=k}
  if($5>=0.40 && !(k in a40)) a40[k]=$3; if($5>=0.45 && !(k in a45)) a45[k]=$3; if($5>=0.50 && !(k in a50)) a50[k]=$3
  if($5>=0.90 && !(k in a90)) a90[k]=$3; if($5>=0.99 && !(k in a99)) a99[k]=$3; if($5>best[k]) best[k]=$5; last[k]=$3; fin[k]=$5 }
  END { printf "%-32s %6s %6s %6s %6s %6s %6s %6s %6s\n","seed arm","→40%","→45%","→50%","→90%","→99%","steps","final","best"
    for(i=1;i<=n;i++){k=order[i]; printf "%-32s %6s %6s %6s %6s %6s %6d %6.3f %6.3f\n",k,(k in a40)?a40[k]:"-",(k in a45)?a45[k]:"-",(k in a50)?a50[k]:"-",(k in a90)?a90[k]:"-",(k in a99)?a99[k]:"-",last[k],fin[k],best[k]} }' curves*.tsv
