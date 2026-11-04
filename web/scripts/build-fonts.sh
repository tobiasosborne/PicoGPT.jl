#!/usr/bin/env bash
# Rebuild the self-hosted webfonts in public/fonts from their upstream sources.
# All four families are SIL Open Font License 1.1; licences are copied alongside.
# Requires: curl, unzip, uv (runs fontTools' pyftsubset with brotli for WOFF2).
set -euo pipefail
OUT="$(cd "$(dirname "$0")/.." && pwd)/src/lib/fonts"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
G=https://raw.githubusercontent.com/google/fonts/main/ofl
mkdir -p "$OUT/licenses"
cd "$TMP"
curl -sfLo Literata.ttf        "$G/literata/Literata%5Bopsz,wght%5D.ttf"
curl -sfLo Literata-Italic.ttf "$G/literata/Literata-Italic%5Bopsz,wght%5D.ttf"
curl -sfLo Archivo.ttf         "$G/archivo/Archivo%5Bwdth,wght%5D.ttf"
curl -sfLo Literata-OFL.txt    "$G/literata/OFL.txt"
curl -sfLo Archivo-OFL.txt     "$G/archivo/OFL.txt"
curl -sfLo xcharter-math.zip   https://mirrors.ctan.org/fonts/xcharter-math.zip
curl -sfLo juliamono.tar.gz    https://github.com/cormullion/juliamono/releases/download/v0.63.2/JuliaMono-ttf.tar.gz
unzip -q xcharter-math.zip && mkdir jm && tar xzf juliamono.tar.gz -C jm

subset() { # in out unicodes
  uv run --quiet --with fonttools --with brotli pyftsubset "$1" --output-file="$2" --flavor=woff2 \
    --unicodes="$3" --layout-features='*' --name-IDs='*' --notdef-outline --no-hinting --desubroutinize
}
TEXT='U+0000-017F,U+0192,U+0218-021B,U+02BC-02DD,U+0300-036F,U+0370-03FF,U+2000-206F,U+2070-209F,U+20AC,U+2100-214F,U+2190-21FF,U+2200-22FF,U+2423,U+FB00-FB06'
LABEL='U+0000-017F,U+02BC-02DD,U+0300-036F,U+2000-206F,U+2070-209F,U+2190-21FF,U+2200-22FF,U+25A0-25FF'
MONO='U+0000-017F,U+02B0-02FF,U+0300-036F,U+0370-03FF,U+1D00-1DBF,U+2000-206F,U+2070-209F,U+2100-214F,U+2190-21FF,U+2200-22FF,U+2300-23FF,U+2423,U+2500-257F,U+25A0-25FF,U+27C0-27EF'
subset Literata.ttf        "$OUT/Literata-var.woff2"        "$TEXT"
subset Literata-Italic.ttf "$OUT/Literata-Italic-var.woff2" "$TEXT"
subset Archivo.ttf         "$OUT/Archivo-var.woff2"         "$LABEL"
subset jm/JuliaMono-Regular.ttf  "$OUT/JuliaMono-Regular.woff2"  "$MONO"
subset jm/JuliaMono-SemiBold.ttf "$OUT/JuliaMono-SemiBold.woff2" "$MONO"
subset jm/JuliaMono-RegularItalic.ttf "$OUT/JuliaMono-Italic.woff2" "$MONO"
# Maths: keep the whole font (the MATH table references glyphs outside any range we could guess).
uv run --quiet --with fonttools --with brotli python - <<PY
from fontTools.ttLib import TTFont
f = TTFont('xcharter-math/XCharter-Math.otf'); f.flavor = 'woff2'; f.save('$OUT/XCharter-Math.woff2')
PY
cp Literata-OFL.txt "$OUT/licenses/Literata-OFL.txt"
cp Archivo-OFL.txt  "$OUT/licenses/Archivo-OFL.txt"
cp jm/LICENSE       "$OUT/licenses/JuliaMono-OFL.txt"
cat > "$OUT/licenses/XCharter-Math-OFL.txt" <<TXT
XCharter-Math.otf (Daniel Flipo, CTAN package xcharter-math) is licensed under the
SIL Open Font License, Version 1.1 — https://openfontlicense.org
TXT
ls -la "$OUT"
