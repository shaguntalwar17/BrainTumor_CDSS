import PageShell from "@/components/PageShell";

export default function AboutPage() {
  return (
    <PageShell
      title="About This Project"
      subtitle="Research prototype for AI-assisted brain MRI analysis, explainability, and longitudinal patient monitoring."
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Project Objective</h2>
          <p className="mt-2 text-sm text-slate-300">
            Build an end-to-end platform for tumor detection, segmentation, classification, explainable AI, patient history tracking, longitudinal
            comparison, and hospital-style PDF report generation.
          </p>
        </section>

        <section className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Core Technologies</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
            <li>FastAPI backend with SQLAlchemy database</li>
            <li>PyTorch + MONAI-ready ML training/evaluation scripts</li>
            <li>Grad-CAM explainability pipeline</li>
            <li>RAG-ready patient-history retrieval using vector indexing</li>
            <li>Next.js + Tailwind + Framer Motion frontend</li>
          </ul>
        </section>

        <section className="glass rounded-2xl p-4 lg:col-span-2">
          <h2 className="text-lg font-semibold text-cyan-200">Important Limitations</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
            <li>Tumor staging is intentionally not predicted due to lack of medically validated stage labels.</li>
            <li>2D scans produce area-based progression estimates; this is not equivalent to true clinical volumetric analysis.</li>
            <li>Model outputs can be uncertain and must be clinically verified.</li>
            <li>This software is for educational/research use only, not direct clinical decision-making.</li>
          </ul>
        </section>
      </div>

      <div className="mt-4 glass rounded-2xl p-4 text-sm text-slate-200">
        <p className="font-semibold text-cyan-300">Project made by Shagun Talwar</p>
      </div>
    </PageShell>
  );
}
