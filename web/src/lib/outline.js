// The edition in force: its outline, title and front matter. vite.config.js aliases
// '#edition' to web/src/lib/editions/book.js (the main book) or to <folder>/edition.js when
// BOOK_EDITION=<folder>. Everything else imports from here, so a change of edition is one
// environment variable.
export { OUTLINE, BOOK, PATHS, FRONT, PRINT } from '#edition';

/** Chapter number as printed: 0 → "0", epilogue → "" */
export const chapterLabel = (c) => (c.unnumbered ? '' : String(c.num));
