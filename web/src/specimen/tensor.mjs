/** A column-major Float64 tensor. All indices in these accessors are zero-based. */
export class Tensor {
  /** @param {number[]} shape @param {Float64Array} [data] */
  constructor(shape, data = new Float64Array(shape.reduce((a, b) => a * b, 1))) {
    if (shape.reduce((a, b) => a * b, 1) !== data.length) throw new RangeError('Tensor shape/data mismatch');
    this.shape = Object.freeze([...shape]);
    this.data = data;
    this._attention = null;
  }
  /** @param {...number} indices @returns {number} */
  get(...indices) {
    if (indices.length !== this.shape.length) throw new RangeError('Wrong number of indices');
    let offset = 0, stride = 1;
    for (let i = 0; i < indices.length; i++) {
      checkIndex(indices[i], this.shape[i]);
      offset += indices[i] * stride;
      stride *= this.shape[i];
    }
    return this.data[offset];
  }
  /** A view, with no copy; flatten the remaining dimensions into token columns.
   * @param {number} t @returns {Float64Array} */
  column(t) {
    const rows = this.shape[0];
    checkIndex(t, this.data.length / rows);
    return this.data.subarray(t * rows, (t + 1) * rows);
  }
  /** Rows are strided: supply out to avoid allocating a copy.
   * @param {number} i @param {Float64Array} [out] @returns {Float64Array} */
  row(i, out = new Float64Array(this.shape[1])) {
    if (this.shape.length !== 2) throw new RangeError('row requires a matrix');
    checkIndex(i, this.shape[0]);
    if (out.length !== this.shape[1]) throw new RangeError('Wrong row length');
    for (let j = 0; j < out.length; j++) out[j] = this.data[i + j * this.shape[0]];
    return out;
  }
  /** Cached matrix view: rows = queries, columns = keys.
   * @param {number} head @param {number} [seq] @returns {Tensor} */
  attention(head, seq = 0) {
    if (this.shape.length !== 4) throw new RangeError('attention requires [T,T,H,B]');
    const [T, , H, B] = this.shape;
    checkIndex(head, H); checkIndex(seq, B);
    if (!this._attention) {
      this._attention = Array.from({ length: H * B }, (_, i) =>
        new Tensor([T, T], this.data.subarray(i * T * T, (i + 1) * T * T)));
    }
    return this._attention[head + H * seq];
  }
  toJSON() { return { shape: this.shape, data: Array.from(this.data) }; }
  clone() { return new Tensor(this.shape, this.data.slice()); }
}

function checkIndex(i, n) {
  if (!Number.isInteger(i) || i < 0 || i >= n) throw new RangeError(`Index ${i} outside [0, ${n})`);
}
