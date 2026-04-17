"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ChatInterface from "../../frontend/components/ChatInterface";
import { decodeJwt, getStoredToken } from "../../lib/api";
import type { JwtPayload } from "../../lib/api";

export default function Page() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      router.replace("/");
      return;
    }

    const decoded: JwtPayload | null = decodeJwt(token);
    if (!decoded || decoded.role === "admin") {
      router.replace(decoded?.role === "admin" ? "/admin" : "/");
      return;
    }

    setReady(true);
  }, [router]);

  if (!ready) {
    return <main className="min-h-screen bg-slate-950" />;
  }

  return <ChatInterface />;
}
