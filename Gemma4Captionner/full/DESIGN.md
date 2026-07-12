# Design

## Product Context

Track 2 Captioner is a competition-control interface for validating a Docker
video captioning agent. Design serves evidence: videos, generated captions,
provider routing, scoring, validation, and remaining blockers.

## Visual Direction

Restrained mission-control product UI. Dark charcoal surfaces, high contrast
text, compact controls, visible status chips, and dense proof panels. The design
should feel operational and trustworthy rather than decorative.

## Color Tokens

```css
:root {
  --bg: #0e1116;
  --surface: #171d25;
  --surface-2: #1f2732;
  --line: #303b48;
  --text: #eef3f8;
  --muted: #a9b4c0;
  --accent: #4fd1b5;
  --info: #78a6ff;
  --warning: #f6b85f;
  --danger: #ff6f6f;
}
```

## Typography

Use a familiar product sans stack for reliability and fast loading:

```css
font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
  "Segoe UI", sans-serif;
```

Rules:

- sentence case for headings and UI labels;
- tabular figures for metrics and runtime values;
- no decorative display font inside controls, cards, or proof panels;
- keep paragraphs below 75 characters where possible.

## Components

- Status pill: compact bordered chip with semantic color text.
- Proof row: label, status, short evidence sentence.
- Clip card: video, source URL, four caption blocks.
- Flow step: clickable stage with active state and explanatory detail.
- Command block: copyable command, clear title, no fake disabled controls.

## Interaction

- All buttons and links need hover, focus, and pressed states.
- Motion stays under 250 ms and only communicates state.
- Videos use native controls for accessibility and trust.
- Do not hide JSON, scores, or provider fallbacks behind decorative animation.

## Anti-patterns

- generic SaaS hero sections;
- purple/blue AI gradients;
- nested cards;
- unexplained metrics;
- claims without nearby proof;
- jokes or copy about protected traits or appearance.
