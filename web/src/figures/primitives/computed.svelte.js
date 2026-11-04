// computed.svelte.js — expensive, deterministic figure data, computed once.
//
// Some figures run the engine thousands of times (a 4000-step training run, a
// finite-difference check of all 1007 parameters, Hessian power iterations). They are
// still pure functions of the specimen, so the numbers are the same everywhere:
//
//   server   runs the job to completion; the figure renders `job.tag`, an inert
//            <script type="application/json"> holding the result, next to its SVG;
//   client   hydrates from that tag (identical numbers, no recomputation). Reached by
//            client-side navigation (no tag on the page), it runs the job in time slices
//            of ~10 ms so the page stays responsive: `job.value` is null until it is
//            done and `job.progress` ∈ [0, 1] meanwhile.
//
// A job is a generator function: `yield` (optionally a progress fraction) wherever a
// pause is safe; `return` the result (plain JSON: arrays, numbers, strings).
// Results are rounded to `digits` significant figures on BOTH sides, so the server's
// markup and the client's state agree exactly.
//
//   const job = computed(`fig-slingshot:${eta}`, function* () { …; yield k / n; …; return out; });
//   {@html job.tag}   {#if job.value} …draw… {:else} …progress… {/if}

const memo = new Map();
const browser = typeof window !== 'undefined';

function roundDeep(x, digits) {
	if (typeof x === 'number') return Number.isFinite(x) ? Number(x.toPrecision(digits)) : x;
	if (Array.isArray(x) || ArrayBuffer.isView(x)) return Array.from(x, (v) => roundDeep(v, digits));
	if (x && typeof x === 'object') return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, roundDeep(v, digits)]));
	return x;
}

const escape = (s) => s.replace(/</g, '\\u003c').replace(/>/g, '\\u003e').replace(/&/g, '\\u0026');
const attr = (s) => s.replace(/[^\w:.-]/g, '_');

function runToEnd(make) {
	const it = make();
	for (;;) { const r = it.next(); if (r.done) return r.value; }
}

/** @template T @param {string} key unique per figure and parameters @param {() => Generator<any, T>} make */
export function computed(key, make, { digits = 8, slice = 10 } = {}) {
	const id = attr(key);
	const job = $state({ value: null, progress: 0, tag: '' });

	const finish = (value) => {
		job.value = value;
		job.progress = 1;
		job.tag = `<script type="application/json" data-pico-computed="${id}">${escape(JSON.stringify(value))}</script>`;
	};

	if (memo.has(id)) { finish(memo.get(id)); return job; }

	if (!browser) {
		const value = roundDeep(runToEnd(make), digits);
		memo.set(id, value);
		finish(value);
		return job;
	}

	const el = document.querySelector(`script[data-pico-computed="${id}"]`);
	if (el) {
		const value = JSON.parse(el.textContent);
		memo.set(id, value);
		finish(value);
		return job;
	}

	// client-side navigation: compute in slices
	const it = make();
	const tick = () => {
		const t0 = performance.now();
		for (;;) {
			const r = it.next();
			if (r.done) {
				const value = roundDeep(r.value, digits);
				memo.set(id, value);
				finish(value);
				return;
			}
			if (typeof r.value === 'number') job.progress = r.value;
			if (performance.now() - t0 > slice) break;
		}
		setTimeout(tick, 0);
	};
	setTimeout(tick, 0);
	return job;
}
