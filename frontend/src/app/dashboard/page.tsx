"use client";

import { useEffect, useMemo, useState } from "react";

import DisclaimerBanner from "@/components/DisclaimerBanner";
import MetricCard from "@/components/MetricCard";
import PageShell from "@/components/PageShell";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { ModelMetric } from "@/lib/types";

export default function DashboardPage() {
  const [patients, setPatients] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [summary, setSummary] = useState<any | null>(null);
  const [latestScan, setLatestScan] = useState<any | null>(null);

  useEffect(() => {
    api.get("/patients").then((res) => setPatients(res.data)).catch(() => setPatients([]));
    api.get("/model-metrics").then((res) => setMetrics(res.data.items ?? [])).catch(() => setMetrics([]));
    api.get("/dashboard/summary").then((res) => setSummary(res.data.summary)).catch(() => setSummary(null));
    const latestScanId = window.localStorage.getItem("latest_scan_id");
    if (latestScanId) {
      api.get(`/scans/${latestScanId}`).then((res) => setLatestScan(res.data)).catch(() => setLatestScan(null));
    }
  }, []);

  const classificationCount = useMemo(() => metrics.filter((m) => m.task_type === "classification").length, [metrics]);
  const segmentationCount = useMemo(() => metrics.filter((m) => m.task_type === "segmentation").length, [metrics]);

  return (
    <PageShell
      title="Analysis Dashboard"
      subtitle="Track system usage, model lineup, and research readiness. Use Upload, History, and Comparison pages for patient-level outputs."
    >
      <div className="mb-4">
        <DisclaimerBanner />
      </div>

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
            <li>Tumor stage is intentionally not predicted due to missing validated stage labels in the dataset.</li>
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
            <MetricCard title="Confidence" value={`${(latestScan.scan.confidence_score * 100).toFixed(2)}%`} tone="cyan" />
            <MetricCard title="Risk Level" value={latestScan.scan.risk_category} tone="amber" />
            <MetricCard title="Runtime" value={(latestScan.runtime_mode ?? "demo").toUpperCase()} tone="emerald" />
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {latestScan.assets?.image_url ? <img src={toAbsoluteUrl(latestScan.assets.image_url) ?? ""} alt="MRI" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.mask_url ? <img src={toAbsoluteUrl(latestScan.assets.mask_url) ?? ""} alt="Mask" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.gradcam_url ? <img src={toAbsoluteUrl(latestScan.assets.gradcam_url) ?? ""} alt="Grad-CAM" className="h-36 w-full rounded-lg object-cover" /> : null}
            {latestScan.assets?.overlay_url ? <img src={toAbsoluteUrl(latestScan.assets.overlay_url) ?? ""} alt="Overlay" className="h-36 w-full rounded-lg object-cover" /> : null}
          </div>
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
