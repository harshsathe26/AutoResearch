# AutoResearch Frontend — Academic Workspace

A simple React + Vite frontend prototype for AutoResearch. It includes three navigation sections: Research, Papers, and Reports.

## Current scope

- Working client-side navigation between the three sections.
- Research question input and validation.
- Responsive layout for desktop and mobile.
- Search/filter interactions for example paper and report entries.
- Explicit preview messaging so the UI does not pretend it is running the Python pipeline.

## Not connected yet

This is a frontend-only prototype. It does not call the AutoResearch Python pipeline, create real research runs, or load actual papers/reports. Example entries are placeholders. The Start Research action intentionally explains that the backend is not connected.

## Run locally

Requires Node.js (prefer Node 20 or 22 LTS for broad Vite compatibility).

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

To check the production build:

```bash
npm run build
```

## Next integration step

Inspect the existing Python backend and expose stable endpoints for starting a run, checking stage status, retrieving papers, and listing/downloading reports. Then replace the clearly marked placeholder data and preview-only submit handler with real API calls. Keep the Gemini/API credentials exclusively on the backend.
