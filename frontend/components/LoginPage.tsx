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
  const [organization, setOrganization] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          organization: decoded?.organization || data.organization || organization,
          role,
        }),
      );

      router.replace(role === "admin" ? "/admin" : "/chat");
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || "Unable to log in";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-root">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .login-root {
          font-family: 'Inter', sans-serif;
          min-height: 100vh;
          background: linear-gradient(-45deg, #020617, #0b1323, #0c1a30, #060f1e);
          background-size: 400% 400%;
          animation: gradientDrift 14s ease infinite;
          display: flex;          align-items: center;
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
          margin-bottom: 1.25rem;
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
          color: white;
          cursor: pointer;
          transition: transform 0.1s, background 0.2s;
          margin-top: 0.5rem;
        }

        .login-btn:hover { background: #0284c7; }
        .login-btn:active { transform: scale(0.98); }
        .login-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .error-box {
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.2);
          color: #f87171;
          padding: 0.75rem 1rem;
          border-radius: 12px;
          margin-bottom: 1.5rem;
          font-size: 0.85rem;
          line-height: 1.4;
        }
      `}</style>

      <div className="login-card">
        <div className="monogram">D1</div>
        <h1 style={{ color: "white", textAlign: "center", margin: "0 0 0.5rem", fontSize: "1.5rem", fontWeight: 700 }}>Welcome Back</h1>
        <p style={{ color: "#94a3b8", textAlign: "center", margin: "0 0 2rem", fontSize: "0.9rem" }}>Log in to your organization dashboard</p>

        {error && <div className="error-box">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div>
            <label className="login-label">Organization</label>
            <input
              className="login-input"
              type="text"
              placeholder="e.g. Acme Corp"
              value={organization}
              onChange={(e) => setOrganization(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="login-label">Username</label>
            <input
              className="login-input"
              type="text"
              placeholder="Your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="login-label">Password</label>
            <input
              className="login-input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button className="login-btn" type="submit" disabled={loading}>
            {loading ? "Logging in..." : "Continue"}
          </button>
        </form>

        <p style={{ color: "#64748b", textAlign: "center", marginTop: "1.5rem", fontSize: "0.85rem" }}>
          Don't have an organization?{" "}
          <button 
            onClick={() => router.push("/signup")}
            style={{ color: "#38bdf8", background: "none", border: "none", padding: 0, cursor: "pointer", fontWeight: 500 }}
          >
            Sign up
          </button>
        </p>
      </div>
    </div>;
}
