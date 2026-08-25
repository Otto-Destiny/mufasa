# Slides

Team-facing slides. Each file is a single self-contained HTML page — open it in a
browser, no build step, no network. They are 16:9 and sized in container units, so
they scale to any window and are legible when shared over a call.

| Sheet | File | Covers |
|---|---|---|
| SE-004 | [scientific-ai-pipeline-infographic.html](./scientific-ai-pipeline-infographic.html) | The four MUFASA phases end to end — data engineering, model engineering, retrieval, application — and where we currently are in each. |
| SE-005 – SE-007 | [data-engineering-update-infographic.html](./data-engineering-update-infographic.html) | The data engineering layer in three sheets: the classification workflow, the corpus funnel and domain mix, then coverage and the decisions that saved rework. |
| SE-008 – SE-011 | [graphrag-explained-infographic.html](./graphrag-explained-infographic.html) | How the GraphRAG layer works, for engineers who have not built one: the full architecture, why this is not a search engine, why a measurement is a node rather than an edge, and one question walked end to end. |

Present SE-004 first for orientation, then the data engineering sheets, then the GraphRAG explainer.

These four are diagrams, not documents — the words on them are labels, and the
explanation is meant to be spoken over them. Resist the urge to add prose.

**SE-008** is the architecture: the offline build pipeline across the top, LadybugDB
in the middle holding a full-text index, a vector index and the property graph, and
the runtime path below — query processing, encoder, hybrid retrieval, RRF fusion,
one-hop expansion, cross-encoder rerank, the LLM, the validator, the answer.

The store blocks are deliberately ordered left-to-right to match the runtime steps
that read them, so every connector drops onto the block it actually touches. Keep
that alignment if you move anything, or the arrows start lying.

**SE-009** is the one to dwell on, because the retrieval layer is easy to mistake for
a search engine and that mistake leads straight to the wrong storage choice. A search
engine ranks documents; this layer returns measured claims with provenance. The store
has to serve text, vectors and traversal at once, or the one-hop step has nothing to
hop across.

Worked figures on SE-009, SE-010 and SE-011 are illustrative placeholders, labelled as
such on the sheets. Do not quote them as results.

## Decks

[MUFASA-EagleTeam-ADTC2026-Briefing-v3.pptx](./MUFASA-EagleTeam-ADTC2026-Briefing-v3.pptx)
— the ADTC 2026 team briefing. PowerPoint, so unlike the HTML sheets it is a binary
blob in Git: every save rewrites the whole file and there is no readable diff. Keep
edits to deliberate versions rather than incremental saves, and bump the `v` in the
filename when the content genuinely changes.

## Editing the HTML sheets

Everything is inline — CSS in a `<style>` block, no scripts, no external fonts or
images. Fonts are system stacks on purpose: a web-font CDN is blocked when these
pages are published, and the fallback is silent, so the slide would quietly render
in Times New Roman on someone else's screen.

Type sizes are in `cqi` (container-inline units) against the slide's own container,
not `px` or `rem`. That is what keeps a slide identical at any window size. If you
change one size, keep the others in proportion or the hierarchy breaks.

Long labels in the pipeline chips carry a `sm` or `xs` class that steps the type
down, so a three-word label wraps without stretching its box out of line with the
rest of the row.
