import PageShell from "@/components/PageShell";

const warningSigns = [
  "Persistent or progressively worsening headaches",
  "New-onset seizures or unexplained episodes of confusion",
  "Visual changes, speech difficulty, or one-sided weakness",
  "Personality/memory changes not explained by stress or sleep loss",
  "Nausea/vomiting with neurologic symptoms"
];

const preventionGuidance = [
  "Maintain regular medical check-ups for persistent neurological symptoms.",
  "Avoid delaying specialist consultation when red-flag symptoms appear.",
  "Control cardiovascular and metabolic risks that can complicate neurological health.",
  "Use protective measures in environments with toxic chemical exposure.",
  "Follow evidence-based guidance from neurologists and oncologists."
];

export default function AwarenessPage() {
  return (
    <PageShell
      title="Brain Tumor Awareness / Prevention"
      subtitle="Educational awareness page for early warning signs, risk awareness, and health guidance. This content is informational and not a substitute for clinical consultation."
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Early Warning Signs</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
            {warningSigns.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Risk Awareness & Prevention</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
            {preventionGuidance.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="glass rounded-2xl p-4 lg:col-span-2">
          <h2 className="text-lg font-semibold text-cyan-200">When to Seek Immediate Medical Evaluation</h2>
          <p className="mt-2 text-sm text-slate-300">
            Emergency care is recommended for sudden seizures, acute neurological deficits, severe altered consciousness, or abrupt worsening headaches.
            MRI and specialist neurological assessment are critical for prompt diagnosis and treatment planning.
          </p>
          <p className="mt-3 text-xs text-slate-400">
            Sources for educational content: NIH/MedlinePlus, NCI, and major academic neuro-oncology references.
          </p>
        </section>

        <section className="glass rounded-2xl p-4 lg:col-span-2">
          <h2 className="text-lg font-semibold text-cyan-200">Educational Visual Reference</h2>
          <img
            src="https://commons.wikimedia.org/wiki/Special:FilePath/MRI%20T2%20Brain%20axial%20image.jpg"
            alt="Axial brain MRI educational reference"
            className="mt-3 h-72 w-full rounded-xl object-cover"
          />
          <p className="mt-2 text-xs text-slate-400">
            Public educational MRI visual (Wikimedia Commons). Pair this with physician guidance for accurate interpretation.
          </p>
        </section>
      </div>
    </PageShell>
  );
}
