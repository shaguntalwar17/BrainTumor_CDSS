"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const primaryNav = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Upload MRI Image" },
  { href: "/history", label: "Patient History" },
  { href: "/comparison", label: "Compare Reports" },
  { href: "/awareness", label: "Brain Tumor Awareness / Prevention" },
  { href: "/reports", label: "Report Download" }
];

const secondaryNav = [
  { href: "/dashboard", label: "Analysis Dashboard" },
  { href: "/models", label: "Model Performance" },
  { href: "/about", label: "About Project" }
];

export default function AppSidebar() {
  const path = usePathname();

  return (
    <>
      <aside className="hidden min-h-screen w-72 border-r border-slate-700/60 bg-slate-950/70 p-5 backdrop-blur-xl lg:block">
        <div className="mb-6">
          <p className="text-xs uppercase tracking-[0.28em] text-cyan-300/80">Medical AI Suite</p>
          <h1 className="mt-2 text-xl font-bold text-slate-100">Brain MRI Intelligence</h1>
          <p className="mt-1 text-xs text-slate-400">AI-assisted research prototype</p>
        </div>

        <nav className="space-y-1">
          {primaryNav.map((item) => {
            const active = path === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`block rounded-xl px-3 py-2.5 text-sm transition ${
                  active
                    ? "border border-cyan-300/30 bg-cyan-400/15 text-cyan-100"
                    : "border border-transparent text-slate-200 hover:border-slate-600 hover:bg-slate-900/70"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="my-5 h-px bg-slate-700/60" />

        <nav className="space-y-1">
          {secondaryNav.map((item) => {
            const active = path === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`block rounded-xl px-3 py-2 text-sm transition ${
                  active ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:bg-slate-900/60 hover:text-slate-200"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-8 rounded-2xl border border-cyan-400/20 bg-cyan-500/10 p-3 text-xs text-cyan-100">
          Project made by Shagun Talwar
        </div>
      </aside>

      <div className="sticky top-0 z-40 border-b border-slate-700/60 bg-slate-950/80 px-3 py-2 backdrop-blur-xl lg:hidden">
        <div className="flex gap-2 overflow-x-auto pb-1">
          {primaryNav.map((item) => {
            const active = path === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs ${
                  active ? "bg-cyan-400/20 text-cyan-100" : "bg-slate-800/80 text-slate-300"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </>
  );
}
