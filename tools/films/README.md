# Films from the living figures

From the repository root:

```sh
node tools/films/render.mjs
node tools/films/render.mjs fig-train-live fig-forward-backward
```

The renderer builds `web/` with `BOOK_STRICT=1`, serves the static build on an
available loopback port, and closes its server and Chromium when finished. It
writes `media/films/<id>.mp4`, `.webm`, and `.png` (the final frame). Existing
outputs for selected IDs are replaced. No screen recording, separate animation
engine, training fixture, or trace generation is involved.

Prerequisites: the dependencies in `web/node_modules` and `print/node_modules`
(`npm ci` in each directory), Pandoc as described in `web/README.md`, Playwright's
Chromium (`cd print && npx playwright install chromium`), and ffmpeg with libx264
and libvpx-vp9. ffmpeg is found in `~/.local/bin/ffmpeg`, then on PATH. This work
used Node 24, Playwright 1.64.1 and its bundled Chromium.

| Figure ID | Duration | First/last holds | What unfolds |
| --- | ---: | ---: | --- |
| `fig-forward-backward` | 12 s | 1.2 s each | Values flow down; covectors return up the same paths. |
| `fig-train-live` | 16 s | 1.5 s each | The seeded specimen learns over 200 real AdamW updates. |
| `fig-attention-triangle` | 10 s | 1.2 s each | Scores → causal mask → row softmax, for both heads. |
| `fig-energy-ladder` | 10 s | 1.2 s each | The live Gibbs distribution cools from τ = 8 to ¼. |

These were chosen because their changing marks explain a computation. The
chapter 3 seven-temperature comparison and the chapter 8 slingshot are static
comparisons; the energy ladder provides the continuous temperature sweep, and
the attention triangle provides the existing timed attention computation.

All films are 1920 × 1080 at 30 fps, on the book's light paper. Chromium renders
a 1280 × 720 CSS viewport at **device scale 2**; ffmpeg downsamples its 2560 ×
1440 PNGs with Lanczos. Each figure keeps its book layout, fitted once to the
frame. Titles use the figure registry; glyphs, labels, colours, equations and
all numbers come from the actual figure component. Film-only styling hides
controls, captions, hover instructions and CSS transitions. No overlays are
composited by the encoder. MP4 uses H.264, CRF 18, yuv420p and faststart; WebM
uses VP9, CRF 32 and yuv420p. There is no audio.

## Determinism and review

Every frame calls and awaits `window.__pico.setT(t)` on
`/figures/<id>/?film&t=0`. The hook now awaits Svelte's update and two animation
frames. No play button is pressed. Fonts must be ready before capture. The
clock is linear between the first and final holds; the native figure maps it
to the computation:

- The forward/backward and attention clocks already existed.
- The energy ladder maps `t` logarithmically from τ = 8 to ¼.
- Live training maps `t` to `round(201**t - 1)` updates, matching the loss
  chart's log step axis. It runs the original seeded engine, regenerates samples
  at its normal checkpoints, and rebuilds the run when seeking backwards.
  Future sample ghosts and the reference loss run are hidden for the film.

The additional figure hooks activate only when `attrs.t` is supplied. Normal
reader interaction and print states retain their own behaviour.

Before encoding each figure, the script seeks to 0.37 → 0.81 → 0.37 and requires
byte-identical screenshots at the repeated time. It checks fixed figure geometry,
hidden interaction controls, and that the training endpoint is step 200; it
rejects browser errors or failed asset requests. Five review
PNGs per figure (t = 0, ¼, ½, ¾, 1) are saved to a temporary directory printed at
the end. Inspect them at full size for labels, fonts, collisions and framing.
They are 2560 × 1440; the delivered posters are 1920 × 1080.

For a fast review without encoding:

```sh
node tools/films/render.mjs --stills
FILM_REVIEW_DIR=/tmp/pico-film-review node tools/films/render.mjs --stills fig-train-live
```

To use a running development or preview server instead of building and serving:

```sh
BOOK_URL=http://127.0.0.1:5173/ node tools/films/render.mjs fig-energy-ladder
```

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `BOOK_URL` | Existing site base URL; skips the build and local static server. |
| `FILM_OUTPUT` | Output directory; default `media/films`. |
| `FILM_REVIEW_DIR` | Persist review PNGs in this directory instead of a new temporary directory. |
| `PLAYWRIGHT_MODULE` | Import path of an alternative Playwright module. |
| `FFMPEG` | ffmpeg executable path. |

For identical pixels, use the same fonts, Playwright/Chromium version and host
rendering environment; browser versions may rasterise text differently. Encoded
bytes may also differ across ffmpeg versions. Timing and engine state do not
depend on capture speed.

Repository checks:

```sh
(cd web && BOOK_STRICT=1 npm run build)
node --test web/src/specimen/
```
