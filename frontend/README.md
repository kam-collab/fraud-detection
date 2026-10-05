# Fraud Analyst Console (frontend)

Next.js console for the payment fraud-detection service in this repository. It reads the
FastAPI scoring API and shows model quality, single-transaction scoring, the manual review
queue and drift monitoring.

**All data is synthetic.** Transactions, devices, merchants and labels come from a seeded
generator; nothing here describes real customers or payments.

## Stack

Next.js 16 (App Router, Turbopack) · React 19 · TypeScript (strict) · Tailwind CSS v4 ·
TanStack Query v5 · Recharts 3 · lucide-react · shadcn-style UI primitives · next-themes.

## Setup

Requires Node.js 20.9 or newer and the API running on port 8000.

```bash
# 1. from the project root: start the API (http://localhost:8000, docs at /docs)
make api

# 2. in another terminal
cd frontend
npm install
cp .env.example .env.local   # optional; the default already points at localhost:8000
npm run dev                  # http://localhost:3000
```

`NEXT_PUBLIC_API_URL` sets the API base URL (default `http://localhost:8000`). It is read at
build time. The browser calls the API directly, so the API must allow the console's origin
through CORS; it allows `http://localhost:3000` and `http://127.0.0.1:3000` by default
(`CORS_ORIGINS` on the API side).

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Development server on port 3000 |
| `npm run build` | Production build |
| `npm run start` | Serve the production build on port 3000 |
| `npm run lint` | ESLint (`eslint-config-next`, core web vitals + TypeScript) |
| `npm run type-check` | `tsc --noEmit` |

## Pages

| Route | Content | API |
| --- | --- | --- |
| `/` | KPI tiles, confusion matrix (review or block threshold), precision-recall curve with both thresholds, champion vs baseline, model selection, fraud rate by month, EDA breakdowns | `GET /api/v1/overview`, `GET /api/v1/model` |
| `/score` | Transaction form with sample loaders; score, band, thresholds, top reasons, explanation and its source; HTTP 422 errors shown per field | `GET /api/v1/transactions/sample`, `POST /api/v1/score`, `GET /api/v1/explain/{id}` |
| `/review` | Paginated queue, detail panel with reasons and explanation, approve/decline, and whether the decision matched the true label | `GET /api/v1/review-queue`, `POST /api/v1/review-queue/{id}/decision`, `GET /api/v1/explain/{id}` |
| `/monitoring` | Month selector with status, alerts and recommended actions, performance trend against the baseline and its 95% CI, business metrics, recall by fraud pattern, score histogram, data-drift table, September remediation | `GET /api/v1/monitoring`, `GET /api/v1/model` |

The header polls `GET /readiness` every 30 seconds and shows whether the API is up.

## Structure

```
src/
  app/
    layout.tsx, globals.css        root layout, design tokens (light and dark)
    (dashboard)/
      layout.tsx                   app shell: side nav, header, footer
      page.tsx                     /            overview
      score/page.tsx               /score
      review/page.tsx              /review
      monitoring/page.tsx          /monitoring
  components/
    ui/                            primitives: button, card, badge, table, input, label, skeleton, segmented
    common/                        page header, KPI tile, status/band badges, error/empty/loading states,
                                   score meter, reasons table, explanation card, theme toggle, API status
    charts/                        Recharts wrappers: chart card (with table view), rate bars, monthly line,
                                   PR curve, score histogram, metric trend, recall by pattern
    dashboards/                    one folder per page: overview, score, review, monitoring
    layouts/                       app shell, side nav, header bar, footer
  hooks/                           TanStack Query hooks and query keys
  lib/
    api/                           typed fetch client (ApiError) and one function per endpoint
    format.ts, features.ts         INR / percentage / date formatting, feature and pattern labels
    providers.tsx                  QueryClient and theme providers
  types/                           API response and request types
```

## Notes

- **Formatting.** Money is INR with Indian digit grouping (`₹28,560`, `₹10.37L`); rates are percentages.
- **Validation.** The scoring form does not duplicate the API's rules. It sends what was typed and
  shows the API's HTTP 422 response, both request-schema errors and data-validation errors.
- **Errors.** Every query has loading, empty and error states. When the API cannot be reached the
  page says so and how to start it (`make api`).
- **Accessibility.** Status and decision band are always icon plus text, never colour alone. Every
  chart has a table view. Form fields are labelled, focus is visible, and there is a skip link.
- **Themes.** Light and dark; follows the system setting, with a toggle in the header.
- **Review decisions** are held in the API's memory and reset when the API restarts.
- **Precision-recall curve.** The chart interpolates precision on a fixed recall grid so both models
  share one x axis and one tooltip. That is for display only; all reported metrics come from the API.
- **LLM notes.** "Ask for LLM-written note" calls `/explain/{id}?llm=true`. The server returns the
  template unless the operator has enabled `EXPLANATIONS_LLM`; the UI says when that happened.
