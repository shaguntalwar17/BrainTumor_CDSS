"use client";

import Link from "next/link";
import { motion } from "framer-motion";

import DisclaimerBanner from "@/components/DisclaimerBanner";

const features = [
  "Tumor Detection (Yes/No)",
  "Tumor Segmentation (Mask + Overlay)",
  "Tumor Type Classification",
  "Grad-CAM Explainability",
  "Patient-wise Scan History",
  "Longitudinal MRI Comparison",
  "RAG-grounded Summary Retrieval",
  "Professional AI-Assisted PDF Reports"
];

export default function HomePage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-10 md:px-6">
      <section className="grid items-center gap-8 md:grid-cols-2">
        <motion.div initial={{ opacity: 0, y: 26 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }} className="space-y-5">
          <p className="inline-flex rounded-full border border-cyan-300/35 bg-cyan-400/10 px-3 py-1 text-xs uppercase tracking-widest text-cyan-200">
            Research Prototype
          </p>
          <h1 className="text-4xl font-bold leading-tight text-slate-100 md:text-5xl">
            AI Platform for Brain MRI Tumor Detection, Segmentation, Explainability, and Longitudinal Tracking
          </h1>
          <p className="text-slate-300">
            End-to-end academic system for MRI upload, tumor analysis, progression comparison across time, and hospital-style report generation.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/upload" className="rounded-xl bg-cyan-400 px-5 py-2 font-semibold text-slate-950 shadow-glow transition hover:bg-cyan-300">
              Start Analysis
            </Link>
            <Link href="/models" className="rounded-xl border border-slate-500 px-5 py-2 font-semibold text-slate-200 hover:bg-slate-800">
              View Model Performance
            </Link>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8 }} className="glass rounded-3xl p-5">
          <div className="grid gap-3">
            {features.map((f, idx) => (
              <motion.div
                key={f}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.12 + idx * 0.05 }}
                className="rounded-lg border border-slate-600/50 bg-slate-900/45 px-3 py-2 text-sm text-slate-200"
              >
                {f}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section className="mt-10 grid gap-4">
        <DisclaimerBanner />
        <div className="glass rounded-2xl px-4 py-3 text-sm text-slate-200">
          <p className="font-semibold text-cyan-300">Project made by Shagun Talwar</p>
          <p className="mt-1">
            This project is intended for educational/research showcase and viva/demo presentation. It is not a replacement for medical diagnosis.
          </p>
        </div>
      </section>
    </main>
  );
}
