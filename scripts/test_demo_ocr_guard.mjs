import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const demoPath = new URL('../Gemma4Captionner/full/docs/demo.html', import.meta.url);
const source = readFileSync(demoPath, 'utf8');
const start = source.indexOf('function editDistance');
const end = source.indexOf('async function repairCaptionQualityV18');
assert.ok(start >= 0 && end > start, 'quality guard functions must exist in demo.html');

const qualityGuard = new Function(`${source.slice(start, end)}; return { hasLexicalCorruption, hasGenericHumour, hasSpeedInversion, hasCaptionQualityRisk };`)();
const evidence = "The display counts down to 00.00, briefly shows 'BUKD3', and changes to 'ERROR' in red.";

assert.equal(qualityGuard.hasLexicalCorruption('The display changes to ERROR.', evidence), false);
assert.equal(qualityGuard.hasLexicalCorruption('The display briefly shows BUKD3.', evidence), false);
assert.equal(qualityGuard.hasLexicalCorruption('The scene contains LLBSSLSB orange text.', evidence), true);
assert.equal(qualityGuard.hasLexicalCorruption('The video is aL a montage.', evidence), true);
assert.equal(qualityGuard.hasLexicalCorruption('The result causes a totalest collapse.', evidence), true);

const trafficEvidence = '- Vehicles travel along a multi-lane road and enter a concrete tunnel.';
assert.equal(qualityGuard.hasGenericHumour('humorous_non_tech', 'The visible sequence arrives like a family photo album, giving each grounded moment an entrance.'), true);
assert.equal(qualityGuard.hasGenericHumour('humorous_non_tech', 'Vehicles enter the tunnel like a family squeezing every suitcase into one car.'), false);

const runningEvidence = '- People run quickly along a paved race route toward the camera.';
assert.equal(qualityGuard.hasSpeedInversion('humorous_tech', runningEvidence, 'The runners move with the processing speed of a dial-up modem loading an image.'), true);
assert.equal(qualityGuard.hasSpeedInversion('humorous_tech', runningEvidence, 'The runners move like a low-latency network racing packets toward the finish.'), false);
assert.equal(qualityGuard.hasSpeedInversion('humorous_tech', runningEvidence, 'The runners move faster than a dial-up modem could load the starting line.'), false);
assert.equal(qualityGuard.hasCaptionQualityRisk(runningEvidence, 'The runners move with the speed of a dial-up modem.', 'humorous_tech'), true);

console.log('demo OCR guard regressions: ok');
