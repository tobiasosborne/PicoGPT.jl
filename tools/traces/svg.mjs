// Plotter-honest SVG, drawn in millimetres at print size (DESIGN.md §9).
// Every mark is a single stroked path; nothing is filled; the paper shows through.
import { writeFileSync } from 'node:fs';

export const ink = { ink: '#181a20', value: '#334994', covector: '#d55339' };
export const stroke = { hairline: 0.10, emphasis: 0.18 };       // mm

const fmt = v => (Math.round(v * 1000) / 1000).toString();
export const polyline = pts => 'M' + pts.map(([x, y]) => fmt(x) + ' ' + fmt(y)).join('L');

const escape = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

export function svg({ width, height, title, desc, paths }) {
  const body = paths.map(({ d, color = ink.ink, width = stroke.hairline, opacity = 1 }) =>
    `<path d="${d}" stroke="${color}" stroke-width="${width}"` +
    (opacity < 1 ? ` stroke-opacity="${fmt(Math.max(0.04, opacity))}"` : '') + '/>');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" ` +
    `width="${width}mm" height="${height}mm" fill="none" stroke-linecap="round" stroke-linejoin="round">` +
    `<title>${escape(title)}</title><desc>${escape(desc)}</desc>` + body.join('') + '</svg>';
}

export const save = (file, s) => (writeFileSync(file, s), console.log(file, (s.length / 1e6).toFixed(2), 'MB'));

// Ramer–Douglas–Peucker: drop points within tol (mm) of the chord.
export function simplify(pts, tol = 0.05) {
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
