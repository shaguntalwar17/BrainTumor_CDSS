import { ReactNode } from "react";

import DisclaimerBanner from "@/components/DisclaimerBanner";

export default function PageShell({ title, subtitle, children }: { title: string; subtitle?: string; children: ReactNode }) {
  return (
    <main className="mx-auto max-w-7xl px-4 py-8 md:px-6">
      <div className="mb-6 space-y-3">
        <h1 className="section-title">{title}</h1>
        {subtitle ? <p className="max-w-4xl text-sm text-slate-300">{subtitle}</p> : null}
        <DisclaimerBanner />
      </div>
      {children}
    </main>
  );
}
