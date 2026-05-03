"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Upload MRI" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/history", label: "Patient History" },
  { href: "/comparison", label: "Comparison" },
  { href: "/models", label: "Model Performance" },
  { href: "/about", label: "About" }
];

export default function Navbar() {
  const path = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-slate-700/60 bg-slate-950/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:px-6">
        <Link href="/" className="text-lg font-bold tracking-wide text-cyan-300">
          Brain MRI Tumor AI
        </Link>
        <nav className="hidden flex-wrap items-center gap-2 md:flex">
          {links.map((link) => {
            const active = path === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full px-3 py-1.5 text-sm transition ${
                  active ? "bg-cyan-400/20 text-cyan-200" : "text-slate-300 hover:bg-slate-800 hover:text-cyan-100"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
