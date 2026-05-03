"use client";

import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";

import DisclaimerBanner from "@/components/DisclaimerBanner";
import MetricCard from "@/components/MetricCard";
import PageShell from "@/components/PageShell";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { UploadScanResult } from "@/lib/types";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadScanResult | null>(null);

  const today = useMemo(() => new Date().toISOString().split("T")[0], []);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) {
      setError("Please upload an MRI file.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData(e.currentTarget);
    form.append("file", file);

    try {
      const res = await api.post<UploadScanResult>("/upload-scan", form, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(res.data);
      window.localStorage.setItem("latest_scan_id", String(res.data.scan_id));
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  const imageUrl = toAbsoluteUrl(result?.image_url);
  const maskUrl = toAbsoluteUrl(result?.mask_url);
  const gradcamUrl = toAbsoluteUrl(result?.gradcam_url);
  const overlayUrl = toAbsoluteUrl(result?.overlay_url);
  const reportUrl = toAbsoluteUrl(result?.report_url);

  return (
    <PageShell
      title="Upload MRI Scan"
      subtitle="Upload MRI with patient details to run detection, segmentation, classification, explainability, longitudinal comparison, and report generation."
    >
      <div className="mb-4">
        <DisclaimerBanner />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <form onSubmit={onSubmit} className="glass rounded-2xl p-5">
          <div className="grid gap-3 md:grid-cols-2">
            <input name="patient_id" required placeholder="Patient ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
            <input name="patient_name" required placeholder="Patient Name" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
            <input name="age" required type="number" min={1} max={120} placeholder="Age" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
            <select name="gender" required className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2">
              <option value="">Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
            <input name="contact" placeholder="Contact (optional)" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
            <input name="scan_date" required type="date" defaultValue={today} className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
          </div>

          <textarea name="doctor_notes" rows={3} placeholder="Radiologist/Doctor notes (optional)" className="mt-3 w-full rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />

          <label className="mt-3 block rounded-lg border border-dashed border-cyan-400/50 bg-cyan-500/5 p-4 text-sm text-slate-200">
            MRI File (JPG/PNG/NIfTI/DICOM)
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.nii,.nii.gz,.dcm"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="mt-2 block w-full"
            />
          </label>

          <button
            type="submit"
            disabled={loading}
            className="mt-4 w-full rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Running AI Analysis..." : "Submit and Analyze"}
          </button>

          {error ? <p className="mt-3 text-sm text-rose-300">{error}</p> : null}
        </form>

        <div className="space-y-4">
          {loading ? (
            <motion.div
              initial={{ opacity: 0.4 }}
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ repeat: Number.POSITIVE_INFINITY, duration: 1.2 }}
              className="glass rounded-2xl p-6"
            >
              <p className="text-lg font-semibold text-cyan-200">Analyzing scan...</p>
              <p className="mt-2 text-sm text-slate-300">Preprocessing, segmentation, classification, Grad-CAM, comparison, and report generation in progress.</p>
            </motion.div>
          ) : null}

          {result ? (
            <div className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <MetricCard title="Tumor Detected" value={result.tumor_detected ? "Yes" : "No"} tone={result.tumor_detected ? "amber" : "emerald"} />
                <MetricCard title="Tumor Type" value={result.tumor_type ?? "N/A"} tone="violet" />
                <MetricCard title="Confidence" value={`${(result.confidence_score * 100).toFixed(2)}%`} tone="cyan" />
                <MetricCard title="Risk Category" value={result.risk_category} tone="amber" />
                <MetricCard title="Tumor Area" value={result.tumor_area.toFixed(2)} subtitle="pixels" tone="emerald" />
                <MetricCard title="Progression" value={result.progression_status} tone="violet" />
                <MetricCard title="Runtime Mode" value={result.runtime_mode.toUpperCase()} tone="cyan" />
                <MetricCard title="Model Version" value={result.model_version} tone="emerald" />
              </div>

              <div className="glass rounded-2xl p-4">
                <p className="text-sm font-semibold text-cyan-200">Class Probability Table</p>
                <div className="mt-2 space-y-1 text-sm text-slate-200">
                  {result.class_probabilities.map((p) => (
                    <div key={p.class_name} className="flex items-center justify-between rounded-lg bg-slate-900/60 px-3 py-1.5">
                      <span>{p.class_name}</span>
                      <span>{(p.probability * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                {imageUrl ? (
                  <div className="glass rounded-2xl p-3">
                    <p className="mb-2 text-sm font-semibold text-cyan-200">MRI</p>
                    <img src={imageUrl} alt="Uploaded MRI" className="h-48 w-full rounded-lg object-cover" />
                  </div>
                ) : null}
                {maskUrl ? (
                  <div className="glass rounded-2xl p-3">
                    <p className="mb-2 text-sm font-semibold text-cyan-200">Segmentation Mask</p>
                    <img src={maskUrl} alt="Predicted mask" className="h-48 w-full rounded-lg object-cover" />
                  </div>
                ) : null}
                {gradcamUrl ? (
                  <div className="glass rounded-2xl p-3">
                    <p className="mb-2 text-sm font-semibold text-cyan-200">Grad-CAM</p>
                    <img src={gradcamUrl} alt="Grad-CAM heatmap" className="h-48 w-full rounded-lg object-cover" />
                  </div>
                ) : null}
                {overlayUrl ? (
                  <div className="glass rounded-2xl p-3">
                    <p className="mb-2 text-sm font-semibold text-cyan-200">Overlay</p>
                    <img src={overlayUrl} alt="Overlay visualization" className="h-48 w-full rounded-lg object-cover" />
                  </div>
                ) : null}
              </div>

              {reportUrl ? (
                <a href={reportUrl} target="_blank" rel="noreferrer" className="inline-flex rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-300">
                  Download PDF Report
                </a>
              ) : null}

              {result.runtime_note ? (
                <div className="rounded-xl border border-cyan-300/40 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-100">
                  {result.runtime_note}
                </div>
              ) : null}

              {result.uncertainty_warning ? (
                <div className="rounded-xl border border-amber-300/50 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
                  {result.uncertainty_warning}
                </div>
              ) : null}
              {result.explainability_warning ? (
                <div className="rounded-xl border border-rose-300/50 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                  {result.explainability_warning}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="glass rounded-2xl p-6 text-sm text-slate-300">
              Submit a scan to view analysis output, MRI/mask/Grad-CAM previews, and report link.
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
