"use client";

import { DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { apiRequest } from "../../lib/api";

type JwtPayload = {
  sub?: string;
  username?: string;
  organization?: string;
  role?: string;
  exp?: number;
};

type UploadResponse = {
  organization: string;
  saved_files: string[];
  rebuilt: boolean;
  message: string;
  drift_summary?: string | null;
};

type DriftReport = {
  organization: string;
  document: string;
  timestamp: string;
  new_chunks: number;
  removed_chunks: number;
  changed_chunks: number;
  unchanged_chunks: number;
  summary: string;
  diffs: Array<{ status: string; old_snippet: string; new_snippet: string; distance: number; source: string }>;
};

type ManagedUser = {
  username: string;
  name: string;
  email: string;
  organization: string;
  role: "admin" | "employee" | string;
};

type AdminDashboardProps = {
  apiBaseUrl?: string;
};

type StatusKind = "success" | "error" | null;

function decodeJwt(token: string): JwtPayload | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
    return JSON.parse(atob(padded)) as JwtPayload;
  } catch {
    return null;
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function AdminDashboard({ apiBaseUrl }: AdminDashboardProps) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const apiRoot = useMemo(
    () => apiBaseUrl ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
    [apiBaseUrl],
  );

  const [token, setToken] = useState<string | null>(null);
  const [profile, setProfile] = useState<JwtPayload | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [statusKind, setStatusKind] = useState<StatusKind>(null);
  const [uploading, setUploading] = useState(false);
  const [driftReport, setDriftReport] = useState<DriftReport | null>(null);
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [userStatus, setUserStatus] = useState<string | null>(null);
  const [userStatusKind, setUserStatusKind] = useState<StatusKind>(null);
  const [newUsername, setNewUsername] = useState("");
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newRole, setNewRole] = useState("employee");
  const [newPassword, setNewPassword] = useState("");
  const [selectedUser, setSelectedUser] = useState("");
  const [editName, setEditName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editRole, setEditRole] = useState("employee");
  const [editPassword, setEditPassword] = useState("");

  useEffect(() => {
    const storedToken = localStorage.getItem("dayone_token");
    if (!storedToken) {
      router.replace("/login");
      return;
    }

    const decoded = decodeJwt(storedToken);
    if (!decoded || decoded.role !== "admin") {
      router.replace(decoded?.role === "employee" ? "/chat" : "/login");
      return;
    }

    setToken(storedToken);
    setProfile(decoded);
  }, [router]);

  useEffect(() => {
    if (!token) return;
    void loadUsers();
  }, [token]);

  useEffect(() => {
    const activeUser = users.find((user) => user.username === selectedUser);
    if (!activeUser) {
      if (users.length > 0) {
        const fallback = users[0];
        setSelectedUser(fallback.username);
        setEditName(fallback.name ?? "");
        setEditEmail(fallback.email ?? "");
        setEditRole(fallback.role ?? "employee");
      } else {
        setSelectedUser("");
        setEditName("");
        setEditEmail("");
        setEditRole("employee");
      }
      return;
    }

    setEditName(activeUser.name ?? "");
    setEditEmail(activeUser.email ?? "");
    setEditRole(activeUser.role ?? "employee");
  }, [selectedUser, users]);

  function signOut() {
    localStorage.removeItem("dayone_token");
    localStorage.removeItem("dayone_profile");
    router.replace("/login");
  }

  function addFiles(incoming: FileList | null) {
    if (!incoming) return;
    const valid = Array.from(incoming).filter((f) => /\.(pdf|csv)$/i.test(f.name));
    if (valid.length > 0) setFiles((curr) => [...curr, ...valid]);
  }

  async function loadUsers() {
    setUsersLoading(true);
    try {
      const data = await apiRequest<ManagedUser[]>({
        url: `${apiRoot}/api/admin/users`,
        method: "GET",
      });
      setUsers(data);
    } catch (error) {
      setUserStatus(error instanceof Error ? error.message : "Failed to load users.");
      setUserStatusKind("error");
    } finally {
      setUsersLoading(false);
    }
  }

  function removeFile(index: number) {
    setFiles((curr) => curr.filter((_, i) => i !== index));
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    addFiles(event.dataTransfer.files);
  }

  async function uploadFiles() {
    if (!token || files.length === 0) return;
    setUploading(true);
    setStatus(null);
    setStatusKind(null);

    try {
      const formData = new FormData();
      const organization = profile?.organization || "";
      formData.append("organization", organization);
      files.forEach((file) => formData.append("files", file));

      const data = await apiRequest<Partial<UploadResponse> & { detail?: string }, FormData>({
        url: `${apiRoot}/api/admin/upload`,
        method: "POST",
        data: formData,
        headers: { "Content-Type": "multipart/form-data" },
      });

      setStatus(data.message || "Upload complete.");
      setStatusKind("success");
      setFiles([]);
      // Fetch full drift report
      try {
        const driftRes = await fetch(`${apiRoot}/api/admin/drift-report`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (driftRes.ok) setDriftReport(await driftRes.json() as DriftReport);
      } catch { /* non-critical */ }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Upload failed.");
      setStatusKind("error");
    } finally {
      setUploading(false);
    }
  }

  async function createUser() {
    setUserStatus(null);
    setUserStatusKind(null);
    try {
      const created = await apiRequest<ManagedUser, Record<string, string>>({
        url: `${apiRoot}/api/admin/users`,
        method: "POST",
        data: {
          username: newUsername,
          name: newName,
          email: newEmail,
          role: newRole,
          password: newPassword,
        },
      });
      setUsers((curr) => [...curr, created].sort((a, b) => a.username.localeCompare(b.username)));
      setNewUsername("");
      setNewName("");
      setNewEmail("");
      setNewRole("employee");
      setNewPassword("");
      setUserStatus(`Created ${created.username}.`);
      setUserStatusKind("success");
      setSelectedUser(created.username);
    } catch (error) {
      setUserStatus(error instanceof Error ? error.message : "Failed to create user.");
      setUserStatusKind("error");
    }
  }

  async function saveUser() {
    if (!selectedUser) return;
    setUserStatus(null);
    setUserStatusKind(null);
    try {
      const updated = await apiRequest<ManagedUser, Record<string, string>>({
        url: `${apiRoot}/api/admin/users/${encodeURIComponent(selectedUser)}`,
        method: "PATCH",
        data: {
          name: editName,
          email: editEmail,
          role: editRole,
          ...(editPassword.trim() ? { password: editPassword } : {}),
        },
      });
      setUsers((curr) => curr.map((user) => (user.username === updated.username ? updated : user)));
      setEditPassword("");
      setUserStatus(`Updated ${updated.username}.`);
      setUserStatusKind("success");
    } catch (error) {
      setUserStatus(error instanceof Error ? error.message : "Failed to update user.");
      setUserStatusKind("error");
    }
  }

  async function deleteUser() {
    if (!selectedUser || selectedUser === profile?.username) return;
    setUserStatus(null);
    setUserStatusKind(null);
    try {
      await fetch(`${apiRoot}/api/admin/users/${encodeURIComponent(selectedUser)}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      }).then(async (response) => {
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error((data as { detail?: string }).detail ?? `HTTP ${response.status}`);
        }
      });
      setUsers((curr) => curr.filter((user) => user.username !== selectedUser));
      setUserStatus(`Deleted ${selectedUser}.`);
      setUserStatusKind("success");
    } catch (error) {
      setUserStatus(error instanceof Error ? error.message : "Failed to delete user.");
      setUserStatusKind("error");
    }
  }

  if (!token) {
    return (
      <main style={{ minHeight: "100vh", background: "#020617", display: "grid", placeItems: "center" }}>
        <div style={{ fontSize: "0.875rem", color: "#475569" }}>Loading admin workspace...</div>
      </main>
    );
  }

  const organization = profile?.organization || "Unknown organization";
  const usernameDisplay = profile?.username || "Admin";
  const avatarInitial = usernameDisplay[0]?.toUpperCase() ?? "A";

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .admin-root {
          font-family: 'Inter', sans-serif;
          min-height: 100vh;
          background: linear-gradient(-45deg, #020617, #0b1323, #0c1a30, #060f1e);
          background-size: 400% 400%;
          animation: gradientDrift 14s ease infinite;
          color: #f1f5f9;
          display: flex;
        }

        @keyframes gradientDrift {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        .admin-sidebar {
          width: 300px;
          flex-shrink: 0;
          border-right: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(10, 18, 35, 0.95);
          backdrop-filter: blur(20px);
          padding: 1.5rem 1.25rem;
          display: flex;
          flex-direction: column;
          min-height: 100vh;
        }

        .user-card {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem;
          border-radius: 16px;
          border: 1px solid rgba(51, 65, 85, 0.6);
          background: rgba(15, 23, 42, 0.7);
          margin-bottom: 1rem;
        }

        .user-avatar {
          width: 38px;
          height: 38px;
          border-radius: 50%;
          background: rgba(56, 189, 248, 0.15);
          border: 1px solid rgba(56, 189, 248, 0.35);
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 700;
          font-size: 0.95rem;
          color: #38bdf8;
          flex-shrink: 0;
        }

        .admin-brand-card {
          border-radius: 16px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(15, 23, 42, 0.7);
          padding: 1.25rem;
          margin-bottom: 1rem;
        }

        .org-badge {
          border-radius: 14px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(2, 6, 23, 0.5);
          padding: 1rem;
          margin-bottom: auto;
        }

        .signout-btn {
          width: 100%;
          border-radius: 14px;
          border: 1px solid rgba(71, 85, 105, 0.6);
          background: rgba(2, 6, 23, 0.5);
          padding: 0.75rem 1rem;
          font-size: 0.875rem;
          font-weight: 500;
          font-family: 'Inter', sans-serif;
          color: #94a3b8;
          cursor: pointer;
          transition: border-color 0.2s, background 0.2s, color 0.2s;
          margin-top: 1rem;
        }

        .signout-btn:hover {
          border-color: rgba(148, 163, 184, 0.5);
          background: rgba(15, 23, 42, 0.7);
          color: #e2e8f0;
        }

        .admin-content {
          flex: 1;
          padding: 2.5rem 2rem;
          overflow-y: auto;
        }

        .upload-card {
          max-width: 760px;
          margin: 0 auto;
          border-radius: 24px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(15, 23, 42, 0.7);
          padding: 2.5rem;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
          backdrop-filter: blur(10px);
        }

        .user-grid {
          margin-top: 1.5rem;
          display: grid;
          grid-template-columns: 1.2fr 1fr;
          gap: 1rem;
        }

        .panel {
          border-radius: 18px;
          border: 1px solid rgba(51, 65, 85, 0.45);
          background: rgba(2, 6, 23, 0.35);
          padding: 1rem;
        }

        .table-wrap {
          overflow: hidden;
          border-radius: 14px;
          border: 1px solid rgba(51, 65, 85, 0.45);
          background: rgba(2, 6, 23, 0.35);
        }

        .user-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.85rem;
        }

        .user-table th,
        .user-table td {
          padding: 0.75rem 0.9rem;
          text-align: left;
          border-bottom: 1px solid rgba(51, 65, 85, 0.35);
        }

        .user-table th {
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #64748b;
          background: rgba(15, 23, 42, 0.75);
        }

        .user-table tr:last-child td {
          border-bottom: none;
        }

        .form-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.85rem;
        }

        .field-stack {
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
        }

        .field-stack label {
          font-size: 0.76rem;
          color: #94a3b8;
          font-weight: 600;
          letter-spacing: 0.03em;
        }

        .field-input,
        .field-select {
          border-radius: 12px;
          border: 1px solid rgba(51, 65, 85, 0.75);
          background: rgba(15, 23, 42, 0.85);
          color: #f8fafc;
          padding: 0.7rem 0.85rem;
          font-size: 0.85rem;
          font-family: 'Inter', sans-serif;
          outline: none;
        }

        .field-input:focus,
        .field-select:focus {
          border-color: rgba(56, 189, 248, 0.5);
          box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
        }

        .role-pill {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          padding: 0.2rem 0.55rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.03em;
        }

        .role-admin {
          color: #fde68a;
          background: rgba(234, 179, 8, 0.12);
          border: 1px solid rgba(234, 179, 8, 0.18);
        }

        .role-employee {
          color: #93c5fd;
          background: rgba(59, 130, 246, 0.12);
          border: 1px solid rgba(59, 130, 246, 0.18);
        }

        .danger-btn {
          border-radius: 14px;
          border: 1px solid rgba(244, 63, 94, 0.3);
          background: rgba(244, 63, 94, 0.1);
          color: #fda4af;
          padding: 0.75rem 1.25rem;
          font-size: 0.875rem;
          font-weight: 600;
          font-family: 'Inter', sans-serif;
          cursor: pointer;
          transition: background 0.2s, border-color 0.2s;
        }

        .danger-btn:hover:not(:disabled) {
          background: rgba(244, 63, 94, 0.16);
          border-color: rgba(244, 63, 94, 0.42);
        }

        .danger-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        @media (max-width: 1024px) {
          .user-grid {
            grid-template-columns: 1fr;
          }
        }

        @keyframes pulseBorder {
          0%, 100% { border-color: rgba(56, 189, 248, 0.5); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.2); }
          50% { border-color: rgba(56, 189, 248, 0.9); box-shadow: 0 0 0 6px rgba(56, 189, 248, 0.08); }
        }

        .dropzone {
          margin-top: 2rem;
          cursor: pointer;
          border-radius: 20px;
          border: 2px dashed rgba(51, 65, 85, 0.7);
          background: rgba(2, 6, 23, 0.4);
          padding: 3.5rem 2rem;
          text-align: center;
          transition: border-color 0.2s, background 0.2s;
        }

        .dropzone:hover {
          border-color: rgba(56, 189, 248, 0.4);
          background: rgba(56, 189, 248, 0.04);
        }

        .dropzone-active {
          animation: pulseBorder 1.2s ease-in-out infinite;
          background: rgba(56, 189, 248, 0.06) !important;
        }

        .dropzone-icon {
          font-size: 2.5rem;
          margin-bottom: 0.75rem;
          display: block;
        }

        .file-list {
          margin-top: 1.5rem;
          border-radius: 16px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(2, 6, 23, 0.4);
          padding: 1rem;
        }

        .file-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          border-radius: 12px;
          border: 1px solid rgba(51, 65, 85, 0.4);
          padding: 0.6rem 0.875rem;
          background: rgba(15, 23, 42, 0.5);
          gap: 0.75rem;
        }

        .file-badge-pdf {
          font-size: 0.65rem;
          font-weight: 700;
          padding: 0.2rem 0.5rem;
          border-radius: 6px;
          background: rgba(244, 63, 94, 0.15);
          color: #fda4af;
          border: 1px solid rgba(244, 63, 94, 0.2);
          flex-shrink: 0;
          letter-spacing: 0.05em;
        }

        .file-badge-csv {
          font-size: 0.65rem;
          font-weight: 700;
          padding: 0.2rem 0.5rem;
          border-radius: 6px;
          background: rgba(34, 197, 94, 0.12);
          color: #86efac;
          border: 1px solid rgba(34, 197, 94, 0.2);
          flex-shrink: 0;
          letter-spacing: 0.05em;
        }

        .file-remove-btn {
          background: none;
          border: none;
          color: #475569;
          cursor: pointer;
          font-size: 1.1rem;
          line-height: 1;
          padding: 0.1rem 0.3rem;
          border-radius: 6px;
          transition: color 0.15s, background 0.15s;
          flex-shrink: 0;
        }

        .file-remove-btn:hover {
          color: #fda4af;
          background: rgba(244, 63, 94, 0.1);
        }

        .status-success {
          border-radius: 14px;
          border: 1px solid rgba(34, 197, 94, 0.25);
          background: rgba(34, 197, 94, 0.1);
          padding: 0.875rem 1rem;
          font-size: 0.875rem;
          color: #86efac;
          margin-top: 1.25rem;
        }

        .status-error {
          border-radius: 14px;
          border: 1px solid rgba(244, 63, 94, 0.25);
          background: rgba(244, 63, 94, 0.1);
          padding: 0.875rem 1rem;
          font-size: 0.875rem;
          color: #fda4af;
          margin-top: 1.25rem;
        }

        .upload-btn {
          border-radius: 14px;
          background: #0ea5e9;
          border: none;
          padding: 0.75rem 1.5rem;
          font-size: 0.875rem;
          font-weight: 600;
          font-family: 'Inter', sans-serif;
          color: #020617;
          cursor: pointer;
          transition: background 0.2s, box-shadow 0.2s;
          box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        }

        .upload-btn:hover:not(:disabled) {
          background: #38bdf8;
          box-shadow: 0 4px 20px rgba(56, 189, 248, 0.4);
        }

        .upload-btn:disabled { opacity: 0.55; cursor: not-allowed; }

        .clear-btn {
          border-radius: 14px;
          border: 1px solid rgba(71, 85, 105, 0.6);
          background: rgba(2, 6, 23, 0.5);
          padding: 0.75rem 1.25rem;
          font-size: 0.875rem;
          font-weight: 500;
          font-family: 'Inter', sans-serif;
          color: #94a3b8;
          cursor: pointer;
          transition: border-color 0.2s, background 0.2s, color 0.2s;
        }

        .clear-btn:hover {
          border-color: rgba(148, 163, 184, 0.5);
          background: rgba(15, 23, 42, 0.7);
          color: #e2e8f0;
        }
      `}</style>

      <main className="admin-root">
        {/* Sidebar */}
        <aside className="admin-sidebar">
          {/* User identity card */}
          <div className="user-card">
            <div className="user-avatar" aria-hidden="true">{avatarInitial}</div>
            <div style={{ minWidth: 0 }}>
              <p style={{ margin: 0, fontSize: "0.875rem", fontWeight: 600, color: "#f1f5f9", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {usernameDisplay}
              </p>
              <p style={{ margin: 0, fontSize: "0.75rem", color: "#64748b" }}>Administrator</p>
            </div>
          </div>

          {/* Brand card */}
          <div className="admin-brand-card">
            <p style={{ margin: 0, fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.2em", color: "#475569", fontWeight: 600 }}>
              Administration
            </p>
            <h1 style={{ margin: "0.4rem 0 0.35rem", fontSize: "1.35rem", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.03em" }}>
              DayOne AI
            </h1>
            <p style={{ margin: 0, fontSize: "0.8rem", color: "#64748b", lineHeight: 1.5 }}>
              Tenant management and policy ingestion.
            </p>
          </div>

          {/* Org badge */}
          <div className="org-badge">
            <p style={{ margin: 0, fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.2em", color: "#475569", fontWeight: 600 }}>
              Organization
            </p>
            <p style={{ margin: "0.4rem 0 0", fontSize: "1rem", fontWeight: 600, color: "#38bdf8" }}>
              {organization}
            </p>
          </div>

          {/* Sign out */}
          <button
            onClick={signOut}
            className="signout-btn"
            aria-label="Sign out of DayOne AI admin"
          >
            Sign out
          </button>
        </aside>

        {/* Main content */}
        <section className="admin-content">
          <div className="upload-card">
            <div>
              <p style={{ margin: 0, fontSize: "0.75rem", color: "#38bdf8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Admin Portal
              </p>
              <h2 style={{ margin: "0.4rem 0 0.5rem", fontSize: "2rem", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.04em" }}>
                Upload HR documents
              </h2>
              <p style={{ margin: 0, fontSize: "0.875rem", color: "#64748b", lineHeight: 1.6 }}>
                Drag and drop PDFs or CSVs to rebuild the tenant FAISS index for{" "}
                <span style={{ color: "#38bdf8" }}>{organization}</span>.
              </p>
            </div>

            {/* Drop zone */}
            <div
              id="admin-dropzone"
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`dropzone${dragActive ? " dropzone-active" : ""}`}
              role="button"
              aria-label="Click or drag files to upload"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click(); }}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.csv"
                onChange={(e) => addFiles(e.target.files)}
                style={{ display: "none" }}
                aria-hidden="true"
              />
              <span className="dropzone-icon">📂</span>
              <p style={{ margin: 0, fontSize: "1.05rem", fontWeight: 600, color: "#e2e8f0" }}>
                Drop files here
              </p>
              <p style={{ margin: "0.35rem 0 0", fontSize: "0.825rem", color: "#475569" }}>
                or click to browse · <span style={{ color: "#64748b" }}>PDF and CSV supported</span>
              </p>
            </div>

            {/* File list */}
            {files.length > 0 ? (
              <div className="file-list">
                <p style={{ margin: "0 0 0.75rem", fontSize: "0.8rem", fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  {files.length} file{files.length > 1 ? "s" : ""} selected
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {files.map((file, index) => (
                    <div key={`${file.name}-${file.lastModified}`} className="file-item">
                      <span className={file.name.toLowerCase().endsWith(".pdf") ? "file-badge-pdf" : "file-badge-csv"}>
                        {file.name.toLowerCase().endsWith(".pdf") ? "PDF" : "CSV"}
                      </span>
                      <span style={{ flex: 1, fontSize: "0.85rem", color: "#cbd5e1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {file.name}
                      </span>
                      <span style={{ fontSize: "0.75rem", color: "#475569", flexShrink: 0 }}>
                        {formatBytes(file.size)}
                      </span>
                      <button
                        onClick={() => removeFile(index)}
                        className="file-remove-btn"
                        aria-label={`Remove ${file.name}`}
                        title="Remove file"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {/* Status message */}
            {status ? (
              <div
                className={statusKind === "error" ? "status-error" : "status-success"}
                role="alert"
              >
                {statusKind === "success" ? "✅ " : "⚠️ "}
                {status}
              </div>
            ) : null}

            {/* Drift report panel */}
            {driftReport && (
              <div style={{ marginTop: "1.5rem", borderRadius: "16px", border: "1px solid rgba(56,189,248,0.2)", background: "rgba(56,189,248,0.04)", padding: "1.25rem" }}>
                <p style={{ margin: "0 0 0.5rem", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.15em", color: "#38bdf8", fontWeight: 700 }}>Policy Change Summary</p>
                <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#e2e8f0", fontWeight: 500 }}>{driftReport.summary}</p>
                <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                  {driftReport.changed_chunks > 0 && (
                    <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.6rem", borderRadius: "20px", background: "rgba(234,179,8,0.12)", color: "#fde68a", border: "1px solid rgba(234,179,8,0.2)" }}>
                      ✏️ {driftReport.changed_chunks} changed
                    </span>
                  )}
                  {driftReport.new_chunks > 0 && (
                    <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.6rem", borderRadius: "20px", background: "rgba(34,197,94,0.12)", color: "#86efac", border: "1px solid rgba(34,197,94,0.2)" }}>
                      ➕ {driftReport.new_chunks} added
                    </span>
                  )}
                  {driftReport.removed_chunks > 0 && (
                    <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.6rem", borderRadius: "20px", background: "rgba(244,63,94,0.12)", color: "#fda4af", border: "1px solid rgba(244,63,94,0.2)" }}>
                      🗑️ {driftReport.removed_chunks} removed
                    </span>
                  )}
                  <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.6rem", borderRadius: "20px", background: "rgba(51,65,85,0.4)", color: "#94a3b8", border: "1px solid rgba(51,65,85,0.4)" }}>
                    ✓ {driftReport.unchanged_chunks} unchanged
                  </span>
                </div>
                {driftReport.diffs.filter(d => d.status === "changed").slice(0, 3).map((diff, i) => (
                  <div key={i} style={{ marginTop: "0.75rem", borderRadius: "10px", border: "1px solid rgba(51,65,85,0.4)", padding: "0.6rem 0.875rem", background: "rgba(2,6,23,0.3)" }}>
                    <p style={{ margin: "0 0 0.3rem", fontSize: "0.7rem", color: "#475569", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>Changed section</p>
                    <p style={{ margin: 0, fontSize: "0.78rem", color: "#94a3b8", lineHeight: 1.5 }}>{diff.new_snippet.slice(0, 160)}…</p>
                  </div>
                ))}
              </div>
            )}

            {/* Action buttons */}
            <div style={{ marginTop: "1.5rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <button
                id="admin-rebuild-btn"
                onClick={uploadFiles}
                disabled={uploading || files.length === 0}
                aria-label="Upload files and rebuild the knowledge index"
                className="upload-btn"
              >
                {uploading ? "Uploading…" : "Rebuild Index"}
              </button>
              <button
                onClick={() => { setFiles([]); setStatus(null); setStatusKind(null); }}
                aria-label="Clear all selected files"
                className="clear-btn"
              >
                Clear selection
              </button>
            </div>
          </div>

          <div className="upload-card" style={{ marginTop: "1.5rem" }}>
            <div>
              <p style={{ margin: 0, fontSize: "0.75rem", color: "#38bdf8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                User Management
              </p>
              <h2 style={{ margin: "0.4rem 0 0.5rem", fontSize: "2rem", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.04em" }}>
                Manage tenant access
              </h2>
              <p style={{ margin: 0, fontSize: "0.875rem", color: "#64748b", lineHeight: 1.6 }}>
                Add, update, and remove users for <span style={{ color: "#38bdf8" }}>{organization}</span> without editing config files.
              </p>
            </div>

            {userStatus ? (
              <div
                className={userStatusKind === "error" ? "status-error" : "status-success"}
                role="alert"
              >
                {userStatusKind === "success" ? "✅ " : "⚠️ "}
                {userStatus}
              </div>
            ) : null}

            <div className="user-grid">
              <div className="panel">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", marginBottom: "0.85rem" }}>
                  <h3 style={{ margin: 0, fontSize: "1rem", color: "#f8fafc" }}>Current users</h3>
                  <span style={{ fontSize: "0.78rem", color: "#64748b" }}>
                    {usersLoading ? "Loading..." : `${users.length} total`}
                  </span>
                </div>

                <div className="table-wrap">
                  <table className="user-table">
                    <thead>
                      <tr>
                        <th>Username</th>
                        <th>Name</th>
                        <th>Role</th>
                      </tr>
                    </thead>
                    <tbody>
                      {users.length === 0 ? (
                        <tr>
                          <td colSpan={3} style={{ color: "#64748b" }}>No users in this organization yet.</td>
                        </tr>
                      ) : (
                        users.map((user) => (
                          <tr
                            key={user.username}
                            onClick={() => setSelectedUser(user.username)}
                            style={{
                              cursor: "pointer",
                              background: user.username === selectedUser ? "rgba(56, 189, 248, 0.08)" : undefined,
                            }}
                          >
                            <td style={{ color: "#f8fafc", fontWeight: 600 }}>{user.username}</td>
                            <td style={{ color: "#cbd5e1" }}>{user.name || "—"}</td>
                            <td>
                              <span className={`role-pill ${user.role === "admin" ? "role-admin" : "role-employee"}`}>
                                {user.role}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div className="panel">
                  <h3 style={{ margin: "0 0 0.85rem", fontSize: "1rem", color: "#f8fafc" }}>Add user</h3>
                  <div className="form-grid">
                    <div className="field-stack">
                      <label htmlFor="new-username">Username</label>
                      <input id="new-username" className="field-input" value={newUsername} onChange={(e) => setNewUsername(e.target.value)} />
                    </div>
                    <div className="field-stack">
                      <label htmlFor="new-role">Role</label>
                      <select id="new-role" className="field-select" value={newRole} onChange={(e) => setNewRole(e.target.value)}>
                        <option value="employee">employee</option>
                        <option value="admin">admin</option>
                      </select>
                    </div>
                    <div className="field-stack">
                      <label htmlFor="new-name">Full name</label>
                      <input id="new-name" className="field-input" value={newName} onChange={(e) => setNewName(e.target.value)} />
                    </div>
                    <div className="field-stack">
                      <label htmlFor="new-email">Email</label>
                      <input id="new-email" className="field-input" type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
                    </div>
                  </div>
                  <div className="field-stack" style={{ marginTop: "0.85rem" }}>
                    <label htmlFor="new-password">Temporary password</label>
                    <input id="new-password" className="field-input" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
                  </div>
                  <div style={{ marginTop: "1rem" }}>
                    <button
                      onClick={() => void createUser()}
                      className="upload-btn"
                      disabled={!newUsername.trim() || !newPassword.trim()}
                    >
                      Create user
                    </button>
                  </div>
                </div>

                <div className="panel">
                  <h3 style={{ margin: "0 0 0.85rem", fontSize: "1rem", color: "#f8fafc" }}>Edit user</h3>
                  <div className="field-stack">
                    <label htmlFor="edit-user">Selected user</label>
                    <select id="edit-user" className="field-select" value={selectedUser} onChange={(e) => setSelectedUser(e.target.value)}>
                      <option value="" disabled>Select a user</option>
                      {users.map((user) => (
                        <option key={user.username} value={user.username}>{user.username}</option>
                      ))}
                    </select>
                  </div>
                  <div className="form-grid" style={{ marginTop: "0.85rem" }}>
                    <div className="field-stack">
                      <label htmlFor="edit-name">Full name</label>
                      <input id="edit-name" className="field-input" value={editName} onChange={(e) => setEditName(e.target.value)} disabled={!selectedUser} />
                    </div>
                    <div className="field-stack">
                      <label htmlFor="edit-role">Role</label>
                      <select id="edit-role" className="field-select" value={editRole} onChange={(e) => setEditRole(e.target.value)} disabled={!selectedUser}>
                        <option value="employee">employee</option>
                        <option value="admin">admin</option>
                      </select>
                    </div>
                    <div className="field-stack">
                      <label htmlFor="edit-email">Email</label>
                      <input id="edit-email" className="field-input" type="email" value={editEmail} onChange={(e) => setEditEmail(e.target.value)} disabled={!selectedUser} />
                    </div>
                    <div className="field-stack">
                      <label htmlFor="edit-password">New password</label>
                      <input id="edit-password" className="field-input" type="password" value={editPassword} onChange={(e) => setEditPassword(e.target.value)} disabled={!selectedUser} />
                    </div>
                  </div>
                  <div style={{ marginTop: "1rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                    <button onClick={() => void saveUser()} className="upload-btn" disabled={!selectedUser}>
                      Save changes
                    </button>
                    <button
                      onClick={() => void deleteUser()}
                      className="danger-btn"
                      disabled={!selectedUser || selectedUser === profile?.username}
                    >
                      Delete user
                    </button>
                  </div>
                  {selectedUser === profile?.username ? (
                    <p style={{ margin: "0.85rem 0 0", fontSize: "0.78rem", color: "#64748b" }}>
                      Your current admin account cannot be deleted from this screen.
                    </p>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
