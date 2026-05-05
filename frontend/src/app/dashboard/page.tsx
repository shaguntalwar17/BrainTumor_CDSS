"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import MetricCard from "@/components/MetricCard";
import PageShell from "@/components/PageShell";
import VolumeSliceViewer from "@/components/VolumeSliceViewer";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { ModelMetric } from "@/lib/types";

export default function DashboardPage() {
  const [patients, setPatients] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [summary, setSummary] = useState<any | null>(null);
  const [latestScan, setLatestScan] = useState<any | null>(null);
  const [correctionFile, setCorrectionFile] = useState<File | null>(null);
  const [correctionNotes, setCorrectionNotes] = useState("");
  const [correctionState, setCorrectionState] = useState<string | null>(null);

  useEffect(() => {
    api.get("/patients").then((res) => setPatients(res.data)).catch(() => setPatients([]));
    api.get("/model-metrics").then((res) => setMetrics(res.data.items ?? [])).catch(() => setMetrics([]));
    api.get("/dashboard/summary").then((res) => setSummary(res.data.summary)).catch(() => setSummary(null));
    const latestScanId = window.localStorage.getItem("latest_scan_id");
    if (latestScanId) {
      api.get(`/scans/${latestScanId}`).then((res) => setLatestScan(res.data)).catch(() => setLatestScan(null));
    }
  }, []);

  async function onMaskCorrectionSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!latestScan?.scan?.id) return;
    if (!correctionFile) {
      setCorrectionState("Please select a corrected mask file first.");
      return;
    }

    const form = new FormData();
    form.append("corrected_mask", correctionFile);
    form.append("correction_notes", correctionNotes);
    form.append("corrected_by", "Radiologist review");

    try {
      const res = await api.post(`/api/scans/${latestScan.scan.id}/correct-mask`, form, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setCorrectionState(res.data?.message ?? "Mask correction saved.");
      const refresh = await api.get(`/scans/${latestScan.scan.id}`);
      setLatestScan(refresh.data);
      setCorrectionFile(null);
      setCorrectionNotes("");
    } catch (err: any) {
      setCorrectionState(err?.response?.data?.detail ?? "Failed to save corrected mask.");
    }
  }

  const classificationCount = useMemo(() => metrics.filter((m) => m.task_type === "classification").length, [metrics]);
  const segmentationCount = useMemo(() => metrics.filter((m) => m.task_type === "segmentation").length, [metrics]);

  return (
    <PageShell
      title="Analysis Dashboard"
      subtitle="Track system usage, model lineup, and research readiness. Use Upload, History, and Comparison pages for patient-level outputs."
    >
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="Registered Patients" value={String(summary?.total_patients ?? patients.length)} tone="cyan" />
        <MetricCard title="Total Scans" value={String(summary?.total_scans ?? 0)} tone="emerald" />
        <MetricCard title="Classification Models" value={String(classificationCount)} tone="violet" />
        <MetricCard title="Segmentation Models" value={String(segmentationCount)} tone="amber" />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <div className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">System Storyboard</h2>
          <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-slate-300">
            <li>Upload MRI and patient details</li>
            <li>Run preprocessing verification and normalization</li>
            <li>Detect tumor and classify tumor type</li>
            <li>Generate segmentation mask and Grad-CAM overlay</li>
            <li>Estimate area/volume and risk category</li>
            <li>Compare with historical scans for progression status</li>
            <li>Store findings, generate PDF report, index summary for RAG</li>
          </ol>
        </div>

        <div className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Medical Limitations</h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-300">
            <li>Stage results are research-only and may rely on proxy rules unless validated stage-labeled model checkpoints are configured.</li>
            <li>2D scans use area-based approximations, not true clinical volumetric measurement.</li>
            <li>Low-confidence predictions must be treated as uncertain outputs.</li>
            <li>All findings require certified radiologist/doctor verification.</li>
          </ul>
        </div>
      </div>

      {latestScan ? (
        <div className="mt-6 glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Latest Analysis Snapshot</h2>
          <div className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard title="Tumor Detected" value={latestScan.scan.tumor_detected ? "Yes" : "No"} tone={latestScan.scan.tumor_detected ? "amber" : "emerald"} />
            <MetricCard title="Tumor Type" value={latestScan.scan.tumor_type ?? "N/A"} tone="violet" />
            <MetricCard title="Stage" value={latestScan.scan.stage_label ?? "N/A"} subtitle={latestScan.scan.stage_method ?? undefined} tone="emerald" />
            <MetricCard title="Confidence" value={`${(latestScan.scan.confidence_score * 100).toFixed(2)}%`} tone="cyan" />
            <MetricCard
              title="Uncertainty"
              value={latestScan.scan.uncertainty_score !== null && latestScan.scan.uncertainty_score !== undefined ? `${(latestScan.scan.uncertainty_score * 100).toFixed(2)}%` : "N/A"}
              subtitle={latestScan.scan.uncertainty_std !== null && latestScan.scan.uncertainty_std !== undefined ? `std ${latestScan.scan.uncertainty_std.toFixed(4)}` : undefined}
              tone="amber"
            />
            <MetricCard title="Risk Level" value={latestScan.scan.risk_category} tone="amber" />
            <MetricCard title="Runtime" value={(latestScan.runtime_mode ?? "demo").toUpperCase()} tone="emerald" />
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {latestScan.assets?.image_url ? <img src={toAbsoluteUrl(latestScan.assets.image_url) ?? ""} alt="MRI" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.mask_url ? <img src={toAbsoluteUrl(latestScan.assets.mask_url) ?? ""} alt="Mask" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.corrected_mask_url ? <img src={toAbsoluteUrl(latestScan.assets.corrected_mask_url) ?? ""} alt="Corrected mask" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.gradcam_url ? <img src={toAbsoluteUrl(latestScan.assets.gradcam_url) ?? ""} alt="Grad-CAM" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.overlay_url ? <img src={toAbsoluteUrl(latestScan.assets.overlay_url) ?? ""} alt="Overlay" className="h-36 w-full rounded-lg object-cover" /> : null}
          </div>
          <div className="mt-3">
            <VolumeSliceViewer
              title="Latest Scan Volume Viewer"
              sliceUrls={latestScan.volume_slice_urls ?? []}
              selectedSliceIndex={latestScan.selected_slice_index}
            />
          </div>
          <form onSubmit={onMaskCorrectionSubmit} className="mt-3 rounded-xl border border-slate-700 bg-slate-900/40 p-3">
            <p className="text-sm font-semibold text-cyan-200">Human-in-the-loop Mask Correction</p>
            <p className="mt-1 text-xs text-slate-300">
              Upload a reviewed mask to store expert corrections for active-learning datasets.
            </p>
            <input
              type="file"
              accept=".png,.jpg,.jpeg"
              onChange={(e) => setCorrectionFile(e.target.files?.[0] ?? null)}
              className="mt-2 block w-full text-sm"
            />
            <textarea
              value={correctionNotes}
              onChange={(e) => setCorrectionNotes(e.target.value)}
              placeholder="Correction notes (optional)"
              className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-900/80 px-3 py-2 text-sm"
            />
            <button className="mt-2 rounded-lg bg-cyan-400 px-3 py-1.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300">
              Save Corrected Mask
            </button>
            {correctionState ? <p className="mt-2 text-xs text-slate-200">{correctionState}</p> : null}
          </form>
          {latestScan.assets?.report_url ? (
            <a href={toAbsoluteUrl(latestScan.assets.report_url) ?? undefined} target="_blank" rel="noreferrer" className="mt-3 inline-flex rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-300">
              Open Latest Report
            </a>
          ) : null}
        </div>
      ) : null}
    </PageShell>
  );
}
