# Frontend repair audit

Preserve mode: Streamlit pages, navigation labels, retrieval, chat persistence and eager startup stay in place.
Design: white surfaces, blue accent, Segoe UI/system sans, 8px controls, 12px content panels. Variance 4, motion 2, density 5.

## Repairs
- Shared CSS instead of conflicting per-page styles; light Streamlit theme.
- Explicit text/placeholder contrast, visible keyboard focus and mobile column collapse.
- Native Streamlit 1.61 chat textarea styling with one outer frame; no fixed textarea height.
- Paper question field and Send button stacked to avoid clipping in the narrow detail column.
- Sidebar Markdown labels constrained to two lines, full title available in tooltip.
- Static HTML rendered with st.html; pagination uses a native keyed container.
- Single-year filter range handled; DOI/arXiv URLs no longer receive duplicate prefixes.
- Reference URLs resolve from abstract metadata or full-text metadata, including externally ingested papers.
- Missing reference URLs stay in an explicit detail-page missing-paper state.

## Verification
- Python syntax checked across src/web.
- AppTest: home, lexical BERT, lexical machine translation, no-match results, chat input.
- AppTest: cold detail URL for actual full-text-only paper 1827454 resolves correctly.
- AppTest: nonexistent detail URL does not render home or retain a stale selected paper.
- Four isolated paper lookup regression tests in tests/test_paper_lookup.py.
- Page tests used real abstract resources and real Chroma metadata, with eager RAG warmup replaced only inside the test harness to avoid rebuilding 204539 chunk BM25 for presentation checks. Production eager startup is unchanged.

## Limits
Screenshots supplied by the user identify the repaired defects. Automated widget tests do not verify browser pixel geometry. Full desktop/mobile visual verification and Lighthouse have not been completed because browser automation was unavailable in the preceding audit.

## Navigation and search continuity
- Search button immediately follows the query field; semantic toggle follows the button.
- Dedicated sticky detail back control, independent of narrow columns.
- Reference and paper links carry an explicit return destination. Chat references carry chat_id; search links carry query, mode, page, survey and year filters.
- Related-paper links inherit the original return destination.
- Shared semantic result cache (30-minute TTL, 64 entries) survives the new Streamlit session created by same-window HTML links. Cache key includes query, store path and collection count. Restarting the server clears it.
- AppTest regression: semantic search -> cold detail URL -> back preserves page 1 and survey/year filters, with exactly one semantic_search call total.
- AppTest regression: chat reference -> detail -> Back to Chat restores the requested conversation and history.
