"use client";

import { FormEvent, useEffect, useState } from "react";

import MetricCard from "@/components/MetricCard";
import PageShell from "@/components/PageShell";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { CompareScansResponse } from "@/lib/types";

export default function ComparisonPage() {
  const [patientIdFromQuery, setPatientIdFromQuery] = useState("");
  const [previousScanFromQuery, setPreviousScanFromQuery] = useState("");
  const [result, setResult] = useState<CompareScansResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    setPatientIdFromQuery(params.get("patient_id") ?? "");
    setPreviousScanFromQuery(params.get("previous_scan_id") ?? "");
  }, []);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    const formData = new FormData(e.currentTarget);
    const payload = {
      patient_id: String(formData.get("patient_id")),
      previous_scan_id: Number(formData.get("previous_scan_id")),
      current_scan_id: Number(formData.get("current_scan_id"))
    };

    try {
      const res = await api.post<CompareScansResponse>("/api/scans/compare", payload);
      setResult(res.data);
    } catch {
      setError("Comparison failed. Check patient/scan IDs.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageShell title="Longitudinal Comparison" subtitle="Compare two scans from the same patient and inspect progression with previous/current visual assets.">
      <form onSubmit={onSubmit} className="glass mb-5 grid gap-3 rounded-2xl p-4 md:grid-cols-4">
        <input name="patient_id" defaultValue={patientIdFromQuery} required placeholder="Patient ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
        <input name="previous_scan_id" defaultValue={previousScanFromQuery} required type="number" placeholder="Previous Scan ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
        <input name="current_scan_id" required type="number" placeholder="Current Scan ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2" />
        <button className="rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-60" disabled={loading}>
          {loading ? "Comparing..." : "Compare"}
        </button>
      </form>

      {error ? <p className="mb-4 text-sm text-rose-300">{error}</p> : null}

      {result ? (
        <div className="space-y-3">
          <a
            href={
              toAbsoluteUrl(
                `/api/reports/comparison/${encodeURIComponent(result.patient_id)}/${result.previous_scan_id}/${result.current_scan_id}`
              ) ?? undefined
            }
            target="_blank"
            rel="noreferrer"
            className="inline-flex rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300"
          >
            Download Comparison PDF
          </a>

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            <MetricCard title="Progression Status" value={result.progression_status} tone="amber" />
            <MetricCard title="Absolute Change" value={result.absolute_change.toFixed(2)} tone="cyan" />
            <MetricCard title="Percentage Change" value={`${result.percentage_change.toFixed(2)}%`} tone="violet" />
            <MetricCard title="Tumor Type Change" value={result.tumor_type_change} tone="emerald" />
            <MetricCard title="Stage Change" value={result.stage_change ?? "N/A"} tone="emerald" />
            <MetricCard title="Confidence Difference" value={result.confidence_difference.toFixed(4)} tone="cyan" />
            <MetricCard title="Risk Level Change" value={result.risk_level_change ?? "N/A"} tone="violet" />
            <MetricCard title="Longitudinal Index" value={result.longitudinal_tumor_progression_index ? `${result.longitudinal_tumor_progression_index.toFixed(2)}/100` : "N/A"} tone="amber" />
          </div>

          <div className="glass rounded-2xl p-4">
            <p className="text-sm font-semibold text-cyan-200">Summary</p>
            <p className="mt-2 text-sm text-slate-200">{result.summary}</p>
            <p className="mt-2 text-xs text-slate-400">
              Previous risk: {result.previous_risk_level ?? "N/A"} | Current risk: {result.current_risk_level ?? "N/A"}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Previous stage: {result.previous_stage_label ?? "N/A"} | Current stage: {result.current_stage_label ?? "N/A"}
            </p>
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm font-semibold text-cyan-200">Previous Scan Assets</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                {result.previous_scan_assets?.image_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.image_url) ?? ""} alt="Previous MRI" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.mask_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.mask_url) ?? ""} alt="Previous mask" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.corrected_mask_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.corrected_mask_url) ?? ""} alt="Previous corrected mask" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.gradcam_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.gradcam_url) ?? ""} alt="Previous gradcam" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.overlay_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.overlay_url) ?? ""} alt="Previous overlay" className="h-32 w-full rounded object-cover" /> : null}
              </div>
            </div>

            <div className="glass rounded-2xl p-4">
              <p className="text-sm font-semibold text-cyan-200">Current Scan Assets</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                {result.current_scan_assets?.image_url ? <img src={toAbsoluteUrl(result.current_scan_assets.image_url) ?? ""} alt="Current MRI" className="h-32 w-full rounded object-cover" /> : null}
                {result.current_scan_assets?.mask_url ? <img src={toAbsoluteUrl(result.current_scan_assets.mask_url) ?? ""} alt="Current mask" className="h-32 w-full rounded object-cover" /> : null}
                {result.current_scan_assets?.corrected_mask_url ? <img src={toAbsoluteUrl(result.current_scan_assets.corrected_mask_url) ?? ""} alt="Current corrected mask" className="h-32 w-full rounded object-cover" /> : null}
                {result.current_scan_assets?.gradcam_url ? <img src={toAbsoluteUrl(result.current_scan_assets.gradcam_url) ?? ""} alt="Current gradcam" className="h-32 w-full rounded object-cover" /> : null}
                {result.current_scan_assets?.overlay_url ? <img src={toAbsoluteUrl(result.current_scan_assets.overlay_url) ?? ""} alt="Current overlay" className="h-32 w-full rounded object-cover" /> : null}
              </div>
            </div>
          </div>

          {result.progression_chart_url ? (
            <div className="glass rounded-2xl p-4">
              <p className="mb-2 text-sm font-semibold text-cyan-200">Tumor Progression Chart</p>
              <img src={toAbsoluteUrl(result.progression_chart_url) ?? ""} alt="Tumor progression chart" className="w-full rounded-lg object-cover" />
            </div>
          ) : null}

          {result.growth_map_url ? (
            <div className="glass rounded-2xl p-4">
              <p className="mb-2 text-sm font-semibold text-cyan-200">Mask Growth/Reduction Map</p>
              <img src={toAbsoluteUrl(result.growth_map_url) ?? ""} alt="Tumor growth change map" className="w-full rounded-lg object-cover" />
              <p className="mt-2 text-xs text-slate-300">
                Red regions indicate newly detected/increased areas, cyan shows stable overlap, and orange regions indicate reduced areas.
              </p>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="glass rounded-2xl p-4 text-sm text-slate-300">No comparison run yet.</div>
      )}
    </PageShell>
  );
}
