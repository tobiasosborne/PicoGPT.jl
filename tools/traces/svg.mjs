// Plotter-honest SVG: every mark is a single stroked path, nothing is filled.
import { writeFileSync } from 'node:fs';

export const ink = { paper: '#f3efe6', ink: '#1b1a17', accent: '#b8462e' };

const fmt = v => (Math.round(v * 100) / 100).toString();
export const polyline = pts => 'M' + pts.map(([x, y]) => fmt(x) + ' ' + fmt(y)).join('L');

export function svg({ width, height, paths, background = ink.paper }) {
  const body = paths.map(({ d, stroke = ink.ink, width = 0.35, opacity = 1 }) =>
    `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-linecap="round" stroke-linejoin="round"/>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">` +
    (background ? `<rect width="100%" height="100%" fill="${background}"/>` : '') +
    body.join('') + '</svg>';
}

export const save = (file, s) => (writeFileSync(file, s), file);

// Ramer–Douglas–Peucker: drop points within tol of the chord. Plotters and browsers both thank us.
export function simplify(pts, tol = 0.15) {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0], [bx, by] = pts.at(-1), dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1;
  let k = 0, dmax = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs(dy * (pts[i][0] - ax) - dx * (pts[i][1] - ay)) / L;
    if (d > dmax) dmax = d, k = i;
  }
  return dmax <= tol ? [pts[0], pts.at(-1)]
                     : [...simplify(pts.slice(0, k + 1), tol).slice(0, -1), ...simplify(pts.slice(k), tol)];
}
