"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { decodeJwt, getStoredToken } from "../lib/api";
import type { JwtPayload } from "../lib/api";

export default function Page() {
  const router = useRouter();

  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      router.replace("/login");
      return;
    }

    const decoded: JwtPayload | null = decodeJwt(token);
    if (!decoded?.role) {
      router.replace("/login");
      return;
    }

    router.replace(decoded.role === "admin" ? "/admin" : "/chat");
  }, [router]);

  return <main className="min-h-screen bg-slate-950" />;
}
