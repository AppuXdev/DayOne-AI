"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { decodeJwt, getStoredToken } from "../lib/api";
import type { JwtPayload } from "../lib/api";

export default function Page() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const token = getStoredToken();
    if (token) {
      const decoded: JwtPayload | null = decodeJwt(token);
      if (decoded?.role) {
        router.replace(decoded.role === "admin" ? "/admin" : "/chat");
        return;
      }
    }
    setIsReady(true);
  }, [router]);

  if (!isReady) {
    return <main className="min-h-screen bg-slate-950" />;
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-50 flex flex-col items-center justify-center relative overflow-hidden">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .landing-root {
          font-family: 'Inter', sans-serif;
        }

        .glow-bg {
          position: absolute;
          inset: 0;
          background: 
            radial-gradient(circle at 20% 30%, rgba(56, 189, 248, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(14, 165, 233, 0.1) 0%, transparent 50%);
          pointer-events: none;
        }

        .hero-title {
          font-size: clamp(2.5rem, 8vw, 4.5rem);
          font-weight: 800;
          letter-spacing: -0.04em;
          line-height: 1.1;
          margin-bottom: 1.5rem;
          background: linear-gradient(to bottom right, #fff, #94a3b8);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
          font-size: clamp(1rem, 3vw, 1.25rem);
          color: #94a3b8;
          max-width: 600px;
          margin-bottom: 3rem;
          line-height: 1.6;
        }

        .cta-btn {
          padding: 1rem 2rem;
          border-radius: 16px;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.2s;
          cursor: pointer;
        }

        .cta-primary {
          background: #0ea5e9;
          color: white;
          border: none;
          box-shadow: 0 10px 30px rgba(14, 165, 233, 0.3);
        }

        .cta-primary:hover {
          background: #38bdf8;
          transform: translateY(-2px);
          box-shadow: 0 15px 40px rgba(56, 189, 248, 0.4);
        }

        .cta-secondary {
          background: rgba(255, 255, 255, 0.05);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.1);
          margin-left: 1rem;
        }

        .cta-secondary:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .monogram-large {
          width: 80px;
          height: 80px;
          border-radius: 24px;
          background: rgba(56, 189, 248, 0.1);
          border: 1px solid rgba(56, 189, 248, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 2rem;
          font-size: 2rem;
          font-weight: 800;
          color: #38bdf8;
          box-shadow: 0 0 40px rgba(56, 189, 248, 0.2);
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .animate-fade-in {
          animation: fadeIn 0.8s ease-out forwards;
        }
      `}</style>

      <div className="glow-bg" />
      
      <div className="landing-root max-w-4xl mx-auto px-6 text-center z-10 animate-fade-in">
        <div className="flex justify-center">
          <div className="monogram-large">D1</div>
        </div>
        
        <h1 className="hero-title">
          Scale your HR <br /> with DayOne AI.
        </h1>
        
        <p className="hero-subtitle">
          The multi-tenant RAG platform for secure, grounded HR document intelligence. 
          Zero-hallucination policy compliance at scale.
        </p>
        
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <button 
            className="cta-btn cta-primary"
            onClick={() => router.push("/signup")}
          >
            Create Organization
          </button>
          <button 
            className="cta-btn cta-secondary"
            onClick={() => router.push("/login")}
          >
            Sign in
          </button>
        </div>
        
        <div className="mt-20 pt-10 border-t border-white/5 flex flex-wrap justify-center gap-x-12 gap-y-6 opacity-40">
          <div className="text-sm font-semibold tracking-widest uppercase">Multi-tenant Isolation</div>
          <div className="text-sm font-semibold tracking-widest uppercase">PGVector Powered</div>
          <div className="text-sm font-semibold tracking-widest uppercase">Verified Grounding</div>
        </div>
      </div>
    </main>
  );
}

