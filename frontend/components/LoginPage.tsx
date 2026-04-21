"use client";

import { FormEvent, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { apiRequest } from "../../lib/api";

type AuthToken = {
  access_token: string;
  token_type: string;
  username: string;
  organization: string;
  role: "admin" | "employee" | string;
  expires_at: string;
};

type JwtPayload = {
  sub?: string;
  username?: string;
  organization?: string;
  role?: string;
  exp?: number;
};

type LoginPageProps = {
  apiBaseUrl?: string;
};

function decodeJwt(token: string): JwtPayload | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
    const json = atob(padded);
    return JSON.parse(json) as JwtPayload;
  } catch {
    return null;
  }
}

export default function LoginPage({ apiBaseUrl }: LoginPageProps) {
  const router = useRouter();
  const apiRoot = useMemo(
    () => apiBaseUrl ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
    [apiBaseUrl],
  );
  const defaultOrganization = process.env.NEXT_PUBLIC_DEMO_ORGANIZATION ?? "org_acme";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [organization, setOrganization] = useState(defaultOrganization);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTestCreds, setShowTestCreds] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await apiRequest<
        Partial<AuthToken> & { detail?: string },
        { username: string; password: string; organization: string }
      >({
        url: `${apiRoot}/auth/login`,
        method: "POST",
        data: { username, password, organization },
      });

      if (!data.access_token) {
        throw new Error("Login succeeded but no access token was returned.");
      }

      localStorage.setItem("dayone_token", data.access_token);
      const decoded = decodeJwt(data.access_token);
      const role = decoded?.role || data.role || "employee";

      localStorage.setItem(
        "dayone_profile",
        JSON.stringify({
          username: decoded?.username || data.username || username,
          organization: decoded?.organization || data.organization || "",
          role,
        }),
      );

      router.replace(role === "admin" ? "/admin" : "/chat");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to log in";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {/* Animated gradient background */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .login-root {
          font-family: 'Inter', sans-serif;
          min-height: 100vh;
          background: linear-gradient(-45deg, #020617, #0b1323, #0c1a30, #060f1e);
          background-size: 400% 400%;
          animation: gradientDrift 14s ease infinite;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 2rem 1rem;
          position: relative;
          overflow: hidden;
        }

        .login-root::before {
          content: '';
          position: absolute;
          inset: 0;
          background:
            radial-gradient(circle at 15% 25%, rgba(56, 189, 248, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 85% 75%, rgba(14, 165, 233, 0.08) 0%, transparent 40%);
          pointer-events: none;
        }

        @keyframes gradientDrift {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        .login-card {
          width: min(420px, 100%);
          border-radius: 24px;
          border: 1px solid rgba(56, 189, 248, 0.15);
          background: rgba(15, 23, 42, 0.85);
          padding: 2.5rem 2rem;
          box-shadow:
            0 0 0 1px rgba(56, 189, 248, 0.08),
            0 25px 60px rgba(2, 6, 23, 0.6);
          backdrop-filter: blur(20px);
          position: relative;
          z-index: 1;
        }

        .monogram {
          width: 56px;
          height: 56px;
          border-radius: 16px;
          background: rgba(56, 189, 248, 0.1);
          border: 1px solid rgba(56, 189, 248, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 1.25rem;
          font-size: 1.25rem;
          font-weight: 700;
          color: #38bdf8;
          letter-spacing: -0.04em;
          box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
        }

        .login-input {
          width: 100%;
          border-radius: 14px;
          border: 1px solid rgba(51, 65, 85, 0.8);
          background: rgba(2, 6, 23, 0.6);
          padding: 0.75rem 1rem;
          color: #f1f5f9;
          font-size: 0.9rem;
          font-family: 'Inter', sans-serif;
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s;
          box-sizing: border-box;
        }

        .login-input::placeholder { color: #475569; }

        .login-input:focus {
          border-color: rgba(56, 189, 248, 0.6);
          box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.12);
        }

        .login-label {
          display: block;
          margin-bottom: 0.5rem;
          font-size: 0.82rem;
          font-weight: 500;
          color: #94a3b8;
          letter-spacing: 0.02em;
        }

        .login-btn {
          width: 100%;
          border-radius: 14px;
          background: #0ea5e9;
          border: none;
          padding: 0.85rem 1rem;
          font-size: 0.9rem;
          font-weight: 600;
          font-family: 'Inter', sans-serif;
          color: #020617;
          cursor: pointer;
          transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        }

        .login-btn:hover:not(:disabled) {
          background: #38bdf8;
          box-shadow: 0 4px 20px rgba(56, 189, 248, 0.4);
        }

        .login-btn:active:not(:disabled) { transform: translateY(1px); }
        .login-btn:disabled { opacity: 0.6; cursor: not-allowed; }

        .error-box {
          border-radius: 12px;
          border: 1px solid rgba(244, 63, 94, 0.25);
          background: rgba(244, 63, 94, 0.1);
          padding: 0.75rem 1rem;
          font-size: 0.85rem;
          color: #fda4af;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(2, 6, 23, 0.3);
          border-top-color: #020617;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          flex-shrink: 0;
        }
      `}</style>

      <main className="login-root">
        <div className="login-card">
          {/* D1 Monogram */}
          <div className="monogram">D1</div>

          <div style={{ textAlign: "center", marginBottom: "2rem" }}>
            <h1 style={{ margin: 0, fontSize: "1.75rem", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.03em" }}>
              DayOne AI
            </h1>
            <p style={{ margin: "0.4rem 0 0", fontSize: "0.875rem", color: "#64748b" }}>
              Secure multi-tenant HR onboarding assistant
            </p>
          </div>

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.1rem" }}>
            <div>
              <label htmlFor="login-username" className="login-label">Username</label>
              <input
                id="login-username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="login-input"
                placeholder="Enter your username"
                autoComplete="username"
                required
              />
            </div>

            <div>
              <label htmlFor="login-organization" className="login-label">Organization</label>
              <input
                id="login-organization"
                value={organization}
                onChange={(e) => setOrganization(e.target.value)}
                className="login-input"
                placeholder="org_acme"
                autoComplete="organization"
                required
              />
            </div>

            <div>
              <label htmlFor="login-password" className="login-label">Password</label>
              <input
                id="login-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type="password"
                className="login-input"
                placeholder="Enter your password"
                autoComplete="current-password"
                required
              />
            </div>

            {error ? (
              <div className="error-box" role="alert">
                {error}
              </div>
            ) : null}

            {showTestCreds ? (
              <div style={{
                borderRadius: "12px",
                border: "1px solid rgba(34, 197, 94, 0.25)",
                background: "rgba(34, 197, 94, 0.1)",
                padding: "1rem",
                fontSize: "0.85rem",
                color: "#86efac"
              }}>
                <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Test Credentials:</div>
                <div style={{ marginBottom: "0.8rem" }}>
                  <strong>Employee:</strong><br />
                  Organization: <code style={{ background: "rgba(0,0,0,0.3)", padding: "0 0.25rem" }}>org_acme</code><br />
                  Username: <code style={{ background: "rgba(0,0,0,0.3)", padding: "0 0.25rem" }}>john_doe</code><br />
                  Password: <code style={{ background: "rgba(0,0,0,0.3)", padding: "0 0.25rem" }}>password123</code>
                </div>
                <div>
                  <strong>Admin:</strong><br />
                  Organization: <code style={{ background: "rgba(0,0,0,0.3)", padding: "0 0.25rem" }}>org_acme</code><br />
                  Username: <code style={{ background: "rgba(0,0,0,0.3)", padding: "0 0.25rem" }}>admin_acme</code><br />
                  (Use "Forgot Password" to reset)
                </div>
              </div>
            ) : null}

            <button
              type="submit"
              id="login-submit-btn"
              disabled={loading}
              aria-label="Sign in to DayOne AI"
              className="login-btn"
              style={{ marginTop: "0.25rem" }}
            >
              {loading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Signing in...
                </>
              ) : "Sign in"}
            </button>
          </form>

          <button
            type="button"
            onClick={() => setShowTestCreds(!showTestCreds)}
            style={{
              width: "100%",
              marginTop: "1.25rem",
              paddingTop: "1.25rem",
              borderTop: "1px solid rgba(51, 65, 85, 0.5)",
              background: "none",
              border: "none",
              color: "#64748b",
              fontSize: "0.8rem",
              cursor: "pointer",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#94a3b8")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}
          >
            {showTestCreds ? "Hide" : "Show"} Test Credentials
          </button>
        </div>
      </main>
    </>
  );
}
