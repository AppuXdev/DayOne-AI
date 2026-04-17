# DayOne AI Frontend

This repo now includes a runnable Next.js App Router frontend at the workspace root.

## Run

```bash
npm install
set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

## Key files

- `app/` — route entrypoints for `/`, `/login`, `/chat`, and `/admin`
- `frontend/components/LoginPage.tsx` — JWT login flow
- `frontend/components/ChatInterface.tsx` — employee chat with SSE streaming and feedback
- `frontend/components/AdminDashboard.tsx` — document uploads, drift summary, and tenant user management
- `lib/api.ts` — shared Axios client and token helpers
