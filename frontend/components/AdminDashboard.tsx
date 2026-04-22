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

type OrgStats = {
  id: string;
  name: string;
  user_count: number;
  document_count: number;
};

type QueryTraceRecord = {
  id: string;
  tenant_id: string;
  query: string;
  trace: any;
  created_at: string;
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

function timeAgo(dateString: string): string {
  const now = new Date();
  const past = new Date(dateString);
  const diff = now.getTime() - past.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'just now';
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
  const [orgStats, setOrgStats] = useState<OrgStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [recentTraces, setRecentTraces] = useState<QueryTraceRecord[]>([]);
  const [tracesLoading, setTracesLoading] = useState(false);
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
  const [minimalMode, setMinimalMode] = useState(false);

  useEffect(() => {
    const isMinimal = localStorage.getItem("dayone_minimal_mode") === "true";
    setMinimalMode(isMinimal);
    if (isMinimal) document.body.classList.add("minimal-mode");
  }, []);

  const toggleMinimalMode = () => {
    const newVal = !minimalMode;
    setMinimalMode(newVal);
    localStorage.setItem("dayone_minimal_mode", String(newVal));
    if (newVal) {
      document.body.classList.add("minimal-mode");
    } else {
      document.body.classList.remove("minimal-mode");
    }
  };

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
    void loadStats();
    void loadTraces();
  }, [token]);

  async function loadStats() {
    setStatsLoading(true);
    try {
      const data = await apiRequest<OrgStats>({
        url: `${apiRoot}/api/org/me`,
        method: "GET",
      });
      setOrgStats(data);
    } catch (error) {
      console.error("Failed to load org stats", error);
    } finally {
      setStatsLoading(false);
    }
  }

  async function loadTraces() {
    setTracesLoading(true);
    try {
      const data = await apiRequest<QueryTraceRecord[]>({
        url: `${apiRoot}/api/admin/traces?limit=5`,
        method: "GET",
      });
      setRecentTraces(data);
    } catch (error) {
      console.error("Failed to load traces", error);
    } finally {
      setTracesLoading(false);
    }
  }

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

        .nav-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem 1rem;
          border-radius: 12px;
          color: #94a3b8;
          text-decoration: none;
          font-size: 0.875rem;
          font-weight: 500;
          transition: all 0.2s;
          margin-bottom: 0.25rem;
          border: 1px solid transparent;
        }

        .nav-item:hover {
          background: rgba(56, 189, 248, 0.05);
          color: #f1f5f9;
        }

        .nav-item-active {
          background: rgba(56, 189, 248, 0.1);
          color: #38bdf8;
          border-color: rgba(56, 189, 248, 0.2);
        }

        .stat-card {
          border-radius: 20px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(15, 23, 42, 0.6);
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
        }

        .stat-card:hover {
          transform: translateY(-4px);
          background: rgba(15, 23, 42, 0.8);
          border-color: rgba(56, 189, 248, 0.3);
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .stat-value {
          font-size: 2rem;
          font-weight: 700;
          color: #f8fafc;
          letter-spacing: -0.02em;
        }

        .stat-label {
          font-size: 0.75rem;
          font-weight: 600;
          color: #64748b;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }

        .activity-card {
          border-radius: 20px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          background: rgba(15, 23, 42, 0.7);
          padding: 1.5rem;
          height: 100%;
          box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
          backdrop-filter: blur(10px);
        }

        .activity-item {
          display: flex;
          gap: 1rem;
          padding: 1.25rem 0;
          border-bottom: 1px solid rgba(51, 65, 85, 0.2);
          transition: all 0.2s ease;
        }

        .activity-item:hover {
          padding-left: 0.5rem;
          background: rgba(56, 189, 248, 0.03);
        }

        .activity-item:last-child {
          border-bottom: none;
        }

        .activity-icon {
          width: 36px;
          height: 36px;
          border-radius: 12px;
          background: rgba(56, 189, 248, 0.1);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #38bdf8;
          font-size: 1rem;
          flex-shrink: 0;
          border: 1px solid rgba(56, 189, 248, 0.2);
        }

        .activity-query {
          font-size: 0.875rem;
          color: #f1f5f9;
          font-weight: 500;
          margin-bottom: 0.25rem;
          display: -webkit-box;
          -webkit-line-clamp: 1;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .activity-meta {
          font-size: 0.75rem;
          color: #64748b;
        }

        .debug-btn {
          width: 100%;
          border-radius: 14px;
          border: 1px solid rgba(56, 189, 248, 0.45);
          background: rgba(56, 189, 248, 0.12);
          padding: 0.75rem 1rem;
          font-size: 0.875rem;
          font-weight: 600;
          font-family: 'Inter', sans-serif;
          color: #7dd3fc;
          cursor: pointer;
          transition: border-color 0.2s, background 0.2s, color 0.2s;
          margin-top: 0.75rem;
        }

        .debug-btn:hover {
          border-color: rgba(56, 189, 248, 0.8);
          background: rgba(56, 189, 248, 0.2);
          color: #bae6fd;
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
          border: 1px solid rgba(51, 65, 85, 0.6);
          background: rgba(15, 23, 42, 0.8);
          padding: 2.5rem;
          box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(12px);
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
          {/* Brand Monogram */}
          <div className="flex items-center gap-3 mb-8 px-2">
            <div className="monogram" style={{ margin: 0, width: "42px", height: "42px", borderRadius: "12px", fontSize: "1rem" }}>D1</div>
            <span style={{ fontWeight: 700, fontSize: "1.1rem", color: "#f8fafc", letterSpacing: "-0.02em" }}>DayOne AI</span>
          </div>

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

          <nav style={{ flex: 1, marginTop: "1rem" }}>
            <div className="nav-item nav-item-active">
              <span>📊</span> Dashboard
            </div>
            <div className="nav-item" onClick={() => router.push("/chat")} style={{ cursor: "pointer" }}>
              <span>💬</span> Chat
            </div>
            <div className="nav-item" onClick={() => router.push("/admin/debug")} style={{ cursor: "pointer" }}>
              <span>🔍</span> Debug Panel
            </div>
          </nav>

          {/* Org badge */}
          <div className="org-badge">
            <p style={{ margin: 0, fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.2em", color: "#475569", fontWeight: 600 }}>
              Organization
            </p>
            <p style={{ margin: "0.4rem 0 0", fontSize: "1rem", fontWeight: 600, color: "#38bdf8" }}>
              {organization}
            </p>
          </div>

          {/* Minimal Mode Toggle */}
          <div style={{ padding: "1rem 0.5rem 0.5rem" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "0.75rem", cursor: "pointer", color: "#64748b", fontSize: "0.8rem" }}>
              <input 
                type="checkbox" 
                checked={minimalMode} 
                onChange={toggleMinimalMode} 
                style={{ width: "16px", height: "16px" }}
              />
              Minimal Mode
            </label>
          </div>

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
          <div style={{ maxWidth: "1000px", margin: "0 auto" }}>
            <header style={{ marginBottom: "2.5rem" }}>
              <h1 style={{ fontSize: "2.25rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.5rem", letterSpacing: "-0.04em" }}>
                Organization Overview
              </h1>
              <p style={{ color: "#94a3b8", fontSize: "1rem" }}>
                Welcome to your <span style={{ color: "#38bdf8", fontWeight: 500 }}>{organization}</span> dashboard.
              </p>
            </header>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1.5rem", marginBottom: "1.5rem" }}>
              <div className="stat-card">
                <span className="stat-label">Total Users</span>
                <span className="stat-value">{statsLoading ? "..." : orgStats?.user_count ?? 0}</span>
                <p style={{ margin: 0, fontSize: "0.75rem", color: "#475569" }}>Active organization members</p>
              </div>
              <div className="stat-card">
                <span className="stat-label">Documents</span>
                <span className="stat-value">{statsLoading ? "..." : orgStats?.document_count ?? 0}</span>
                <p style={{ margin: 0, fontSize: "0.75rem", color: "#475569" }}>Processed HR files</p>
              </div>
              <div className="stat-card">
                <span className="stat-label">System Status</span>
                <span className="stat-value" style={{ color: "#86efac", fontSize: "1.5rem" }}>Operational</span>
                <p style={{ margin: 0, fontSize: "0.75rem", color: "#475569" }}>All services healthy</p>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "1.5rem", marginBottom: "3rem", alignItems: "start" }}>
              <div className="upload-card" style={{ maxWidth: "none", margin: 0 }}>
                <div>
                  <p style={{ margin: 0, fontSize: "0.75rem", color: "#38bdf8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                    Knowledge Base
                  </p>
                  <h2 style={{ margin: "0.4rem 0 0.5rem", fontSize: "2rem", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.04em" }}>
                    Upload HR documents
                  </h2>
                  <p style={{ margin: 0, fontSize: "0.875rem", color: "#64748b", lineHeight: 1.6 }}>
                    Drag and drop PDFs or CSVs to refresh tenant pgvector embeddings.
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

              <div className="activity-card">
                <h3 className="stat-label" style={{ marginBottom: "1.25rem", display: "block" }}>Recent Queries</h3>
                {tracesLoading ? (
                  <p style={{ color: "#64748b", fontSize: "0.875rem" }}>Loading activity...</p>
                ) : recentTraces.length === 0 ? (
                  <p style={{ color: "#64748b", fontSize: "0.875rem" }}>No recent activity.</p>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    {recentTraces.map((trace) => (
                      <div key={trace.id} className="activity-item">
                        <div className="activity-icon">
                          {trace.trace?.abstained ? "🚫" : "💬"}
                        </div>
                        <div style={{ minWidth: 0 }}>
                          <p className="activity-query">{trace.query}</p>
                          <div className="activity-meta">
                            <span>{timeAgo(trace.created_at)}</span>
                            <span style={{ margin: "0 0.5rem" }}>·</span>
                            <span style={{ color: trace.trace?.confidence >= 0.7 ? "#86efac" : trace.trace?.confidence >= 0.4 ? "#fde68a" : "#fda4af" }}>
                              {(trace.trace?.confidence * 100 || 0).toFixed(0)}% match
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <button 
                  onClick={() => router.push("/admin/debug")}
                  style={{ width: "100%", marginTop: "1.5rem", padding: "0.75rem", borderRadius: "12px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "#94a3b8", fontSize: "0.825rem", cursor: "pointer" }}
                >
                  View all traces
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
        </div>
      </section>
    </main>
  </>
);
}
