"use client";

import Link from "next/link";
import { motion } from "framer-motion";

import DisclaimerBanner from "@/components/DisclaimerBanner";

const highlights = [
  "Tumor Detection",
  "Tumor Segmentation Overlay",
  "Tumor Type Classification",
  "Research-Stage Estimation",
  "Explainable AI (Grad-CAM)",
  "Patient Tracking & Longitudinal Comparison",
  "Hospital-Style PDF Reports"
];

const anatomyFacts = [
  {
    title: "Frontal Lobe",
    body: "Supports executive functions, motor planning, and behavioral control. Lesions may alter personality and decision-making."
  },
  {
    title: "Temporal Lobe",
    body: "Important for memory and language comprehension; temporal masses can present with seizures, aphasia, or auditory disturbances."
  },
  {
    title: "Parietal/Occipital Regions",
    body: "Linked to sensory integration and vision pathways. Tumors here may produce visual field changes and spatial disorientation."
  }
];

export default function HomePage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-8 md:px-8">
      <section className="grid items-center gap-8 lg:grid-cols-[1.25fr_1fr]">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65 }}
          className="space-y-5"
        >
          <p className="inline-flex rounded-full border border-cyan-300/45 bg-cyan-500/10 px-3 py-1 text-xs uppercase tracking-[0.25em] text-cyan-200">
            Advanced Medical Imaging AI
          </p>
          <h1 className="text-4xl font-bold leading-tight text-slate-100 md:text-5xl">
            Explainable AI for Brain Tumor Detection, Segmentation, Classification, and Stage Analysis
          </h1>
          <p className="max-w-3xl text-slate-300">
            Professional research platform for MRI-driven tumor analytics with visual explainability, patient tracking, longitudinal comparison,
            and hospital-style downloadable reporting for demo and viva presentations.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/upload" className="rounded-xl bg-cyan-400 px-6 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300">
              Upload MRI Image
            </Link>
            <Link href="/reports" className="rounded-xl border border-slate-500 px-6 py-2.5 text-sm font-semibold text-slate-200 hover:bg-slate-800">
              Report Download
            </Link>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7 }}
          className="glass rounded-3xl p-5"
        >
          <p className="mb-3 text-xs uppercase tracking-[0.2em] text-cyan-200">Platform Scope</p>
          <div className="grid gap-2">
            {highlights.map((item, idx) => (
              <motion.div
                key={item}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + idx * 0.05 }}
                className="rounded-xl border border-slate-700/70 bg-slate-900/45 px-3 py-2 text-sm text-slate-200"
              >
                {item}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section className="mt-8">
        <DisclaimerBanner />
      </section>

      <section className="mt-10 grid gap-4 lg:grid-cols-3">
        {anatomyFacts.map((fact) => (
          <div key={fact.title} className="glass rounded-2xl p-4">
            <h2 className="text-lg font-semibold text-cyan-200">{fact.title}</h2>
            <p className="mt-2 text-sm text-slate-300">{fact.body}</p>
          </div>
        ))}
      </section>

      <section className="mt-8 grid gap-4 lg:grid-cols-2">
        <div className="glass rounded-2xl p-4">
          <h3 className="text-lg font-semibold text-cyan-200">Brain Anatomy Reference</h3>
          <p className="mt-1 text-sm text-slate-300">
            Educational anatomy visual for orientation while interpreting MRI slices and lesion localization.
          </p>
          <img
            src="https://commons.wikimedia.org/wiki/Special:FilePath/NIA%20human%20brain%20drawing.jpg"
            alt="Human brain anatomy illustration"
            className="mt-3 h-64 w-full rounded-xl object-cover"
          />
          <p className="mt-2 text-xs text-slate-400">Source: U.S. public-domain brain anatomy illustration (via Wikimedia Commons).</p>
        </div>
        <div className="glass rounded-2xl p-4">
          <h3 className="text-lg font-semibold text-cyan-200">MRI Structural Example</h3>
          <p className="mt-1 text-sm text-slate-300">
            MRI example visualization for educational context in tumor boundary identification and tissue contrast interpretation.
          </p>
          <img
            src="https://commons.wikimedia.org/wiki/Special:FilePath/MRI_brain.jpg"
            alt="Brain MRI example"
            className="mt-3 h-64 w-full rounded-xl object-cover"
          />
          <p className="mt-2 text-xs text-slate-400">Source: Public-domain MRI image (via Wikimedia Commons).</p>
        </div>
      </section>

      <section className="mt-8 grid gap-4 lg:grid-cols-2">
        <div className="glass rounded-2xl p-4">
          <h3 className="text-lg font-semibold text-cyan-200">Clinical Workflow Story</h3>
          <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-slate-300">
            <li>Upload MRI with patient demographics</li>
            <li>Run detection, segmentation, classification, stage estimation, and explainability</li>
            <li>Store scan in patient timeline with longitudinal metrics</li>
            <li>Generate hospital-style PDF reports and comparison reports</li>
          </ol>
        </div>
        <div className="glass rounded-2xl p-4">
          <h3 className="text-lg font-semibold text-cyan-200">Explore More</h3>
          <div className="mt-2 flex flex-wrap gap-2 text-sm">
            <Link href="/awareness" className="rounded-lg border border-slate-600 px-3 py-1.5 text-slate-200 hover:border-cyan-300 hover:text-cyan-200">
              Brain Tumor Awareness / Prevention
            </Link>
            <Link href="/comparison" className="rounded-lg border border-slate-600 px-3 py-1.5 text-slate-200 hover:border-cyan-300 hover:text-cyan-200">
              Compare Reports
            </Link>
            <Link href="/history" className="rounded-lg border border-slate-600 px-3 py-1.5 text-slate-200 hover:border-cyan-300 hover:text-cyan-200">
              Patient History
            </Link>
          </div>
          <p className="mt-3 text-xs text-slate-400">Project made by Shagun Talwar</p>
        </div>
      </section>
    </main>
  );
}
