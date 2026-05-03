"use client";

import { FormEvent, useState } from "react";
import { useSearchParams } from "next/navigation";

import DisclaimerBanner from "@/components/DisclaimerBanner";
import MetricCard from "@/components/MetricCard";
import PageShell from "@/components/PageShell";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { CompareScansResponse } from "@/lib/types";

export default function ComparisonPage() {
  const search = useSearchParams();
  const patientIdFromQuery = search.get("patient_id") ?? "";
  const previousScanFromQuery = search.get("previous_scan_id") ?? "";
  const [result, setResult] = useState<CompareScansResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
      const res = await api.post<CompareScansResponse>("/compare-scans", payload);
      setResult(res.data);
    } catch {
      setError("Comparison failed. Check patient/scan IDs.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageShell title="Longitudinal Comparison" subtitle="Compare two scans from the same patient and inspect progression with previous/current visual assets.">
      <div className="mb-4">
        <DisclaimerBanner />
      </div>

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
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            <MetricCard title="Progression Status" value={result.progression_status} tone="amber" />
            <MetricCard title="Absolute Change" value={result.absolute_change.toFixed(2)} tone="cyan" />
            <MetricCard title="Percentage Change" value={`${result.percentage_change.toFixed(2)}%`} tone="violet" />
            <MetricCard title="Tumor Type Change" value={result.tumor_type_change} tone="emerald" />
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
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm font-semibold text-cyan-200">Previous Scan Assets</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                {result.previous_scan_assets?.image_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.image_url) ?? ""} alt="Previous MRI" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.mask_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.mask_url) ?? ""} alt="Previous mask" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.gradcam_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.gradcam_url) ?? ""} alt="Previous gradcam" className="h-32 w-full rounded object-cover" /> : null}
                {result.previous_scan_assets?.overlay_url ? <img src={toAbsoluteUrl(result.previous_scan_assets.overlay_url) ?? ""} alt="Previous overlay" className="h-32 w-full rounded object-cover" /> : null}
              </div>
            </div>

            <div className="glass rounded-2xl p-4">
              <p className="text-sm font-semibold text-cyan-200">Current Scan Assets</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                {result.current_scan_assets?.image_url ? <img src={toAbsoluteUrl(result.current_scan_assets.image_url) ?? ""} alt="Current MRI" className="h-32 w-full rounded object-cover" /> : null}
                {result.current_scan_assets?.mask_url ? <img src={toAbsoluteUrl(result.current_scan_assets.mask_url) ?? ""} alt="Current mask" className="h-32 w-full rounded object-cover" /> : null}
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
        </div>
      ) : (
        <div className="glass rounded-2xl p-4 text-sm text-slate-300">No comparison run yet.</div>
      )}
    </PageShell>
  );
}
