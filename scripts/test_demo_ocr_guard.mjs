import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const demoPath = new URL('../Gemma4Captionner/full/docs/demo.html', import.meta.url);
const source = readFileSync(demoPath, 'utf8');
const start = source.indexOf('function editDistance');
const end = source.indexOf('function hasUnsupportedAbsolute');
assert.ok(start >= 0 && end > start, 'quality guard functions must exist in demo.html');

const qualityGuard = new Function(`${source.slice(start, end)}; return { hasLexicalCorruption };`)();
const evidence = "The display counts down to 00.00, briefly shows 'BUKD3', and changes to 'ERROR' in red.";

assert.equal(qualityGuard.hasLexicalCorruption('The display changes to ERROR.', evidence), false);
assert.equal(qualityGuard.hasLexicalCorruption('The display briefly shows BUKD3.', evidence), false);
assert.equal(qualityGuard.hasLexicalCorruption('The scene contains LLBSSLSB orange text.', evidence), true);
assert.equal(qualityGuard.hasLexicalCorruption('The video is aL a montage.', evidence), true);
assert.equal(qualityGuard.hasLexicalCorruption('The result causes a totalest collapse.', evidence), true);

console.log('demo OCR guard regressions: ok');
