# Express prototype

`npm install && npm run build` produces `dist/` with the panel, sandbox module, and manifest. `npm run dev` serves a browser preview on port 5241. Use Adobe's add-on development workflow with a trusted HTTPS server for sideloading the built add-on. The API URL can be changed in the panel; an HTTPS panel may require a trusted HTTPS engine endpoint (uvicorn supports `--ssl-keyfile` and `--ssl-certfile`).

Upload structured task JSON and choose dimensions. Generation uses the main engine. In an Express panel, supported results can be applied as editable text, images, rectangles, and ellipses on a new page. Browser mode supports previews only.

`npm test` runs the pure adapter contract tests. The server and add-on share `src/capabilities.json`; `GET /api/express/capabilities` exposes the declared profile. Apply checks engine validity, profile compatibility, all element properties, then sandbox font/image availability. Unknown properties block application. Changing the selected source/candidate during resource loading prevents stale application.

This is a bounded prototype: it does not extract arbitrary Express documents. Groups, rich text, letter spacing, strokes, rounded corners, automatic capitalization, line caps, and custom line height block Apply rather than being silently lost. Supported plain text includes normal/bold/italic font lookup, left/center/right alignment, color and opacity; basic shapes and prepared original-image PNGs remain individually editable. Equal-z stacking follows document order. Prepared images preserve rendered fit, but original crop controls are not preserved; canonical asset references remain unchanged outside Express.

Font styles use SDK PostScript lookup and reject unavailable/edit-restricted fonts. Geometry is pixels; the SDK takes font sizes in points. The historical 0.75 px-to-point factor is now explicit in the profile, **not host-calibrated**. Host font metrics, line wrapping, default leading, rotation pivots and pixel parity remain unverified. Applying is an experimental editable transfer, not proof of canonical layout fidelity.

All resources are loaded before document mutation. SDK edits occur in `keepContentActiveDuringAsync`'s synchronous follow-up. If an SDK operation fails, the adapter attempts to remove only its newly created page. If cleanup fails, the error directs the user to Undo; no atomicity is promised. The successful result includes a canonical-ID/host-node-ID map scoped to the created page, and actual host text/local-bound observations; no invented metadata is written into the document. The panel retains this report as `window.chameleonLastApplication` for developer inspection.

Build and contract-test success do not establish live Adobe-hosted compatibility. No live host was available during this implementation. Before promotion, use a supported fixture with a plain `A😀B` headline, italic font, centered text, translucent background, one rotated rectangle and one original image. Record host/SDK versions, compare measured font size/wrapping/geometry against the engine's exact PNG, verify each element remains editable, verify stacking, and test failure recovery. Keep `host_verified=false` until that evidence is recorded. Grouped/rich/stroked fixtures should fail preflight before creating a page.

Adobe references checked 2026-10-25:

- [Character style units](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/interfaces/character-styles-input)
- [Editable font availability](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/fonts)
- [Async resource application and synchronous mutation](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/editor)
- [Page removal](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/page-list)

API references: https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/editor and https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/node .
