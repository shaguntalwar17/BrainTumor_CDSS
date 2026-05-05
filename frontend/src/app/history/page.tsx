"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip
} from "chart.js";

import PageShell from "@/components/PageShell";
import VolumeSliceViewer from "@/components/VolumeSliceViewer";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { PatientProfilePayload } from "@/lib/types";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

export default function HistoryPage() {
  const [patientId, setPatientId] = useState("");
  const [profile, setProfile] = useState<PatientProfilePayload | null>(null);
  const [patients, setPatients] = useState<Array<{ patient_id: string; name: string }>>([]);
  const [error, setError] = useState<string | null>(null);
  const [ragQuery, setRagQuery] = useState("How has tumor size changed over time?");
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState<string | null>(null);
  const [ragAnswer, setRagAnswer] = useState<string | null>(null);
  const [ragCitations, setRagCitations] = useState<string[]>([]);

  useEffect(() => {
    api
      .get("/patients")
      .then((res) => setPatients(res.data.map((p: any) => ({ patient_id: p.patient_id, name: p.name }))))
      .catch(() => setPatients([]));
  }, []);

  async function onSearch(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setProfile(null);
    setRagError(null);
    setRagAnswer(null);
    setRagCitations([]);
    try {
      const res = await api.get<PatientProfilePayload>(`/patients/${patientId}`);
      setProfile(res.data);
    } catch {
      setError("Patient not found or backend unavailable.");
    }
  }

  async function onRagSubmit(e: FormEvent) {
    e.preventDefault();
    if (!patientId.trim()) {
      setRagError("Enter a patient ID first.");
      return;
    }

    setRagLoading(true);
    setRagError(null);
    setRagAnswer(null);
    setRagCitations([]);

    try {
      const res = await api.post("/api/rag/query", {
        patient_id: patientId.trim(),
        query: ragQuery,
        top_k: 5
      });
      setRagAnswer(res.data?.grounded_answer ?? "No grounded answer generated.");
      setRagCitations(Array.isArray(res.data?.citations) ? res.data.citations : []);
    } catch {
      setRagError("Unable to generate a grounded history summary right now.");
    } finally {
      setRagLoading(false);
    }
  }

  const chartData = useMemo(() => {
    if (!profile?.scans?.length) return null;
    const labels = profile.scans.map((s) => s.scan.scan_date);
    const values = profile.scans.map((s) => s.scan.tumor_volume ?? s.scan.tumor_area);
    return {
      labels,
      datasets: [
        {
          label: "Tumor Area/Volume Trend",
          data: values,
          borderColor: "#22D3EE",
          backgroundColor: "rgba(34, 211, 238, 0.2)",
          tension: 0.35,
          fill: true
        }
      ]
    };
  }, [profile]);

  const latestVolumeSlices = useMemo(() => {
    if (!profile?.scans?.length) return [];
    const latest = [...profile.scans].sort((a, b) => a.scan.scan_date.localeCompare(b.scan.scan_date)).at(-1);
    return latest?.volume_slice_urls ?? [];
  }, [profile]);

  const latestSelectedSlice = useMemo(() => {
    if (!profile?.scans?.length) return null;
    const latest = [...profile.scans].sort((a, b) => a.scan.scan_date.localeCompare(b.scan.scan_date)).at(-1);
    return latest?.selected_slice_index ?? null;
  }, [profile]);

  return (
    <PageShell title="Patient History" subtitle="Search patient records, review scan timeline, and inspect historical MRI/mask/Grad-CAM/report assets.">
      <form onSubmit={onSearch} className="glass mb-4 flex flex-wrap gap-3 rounded-2xl p-4">
        <input
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="Enter Patient ID"
          className="flex-1 rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2"
        />
        <button className="rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-300">Search</button>
      </form>

      {patients.length ? (
        <div className="mb-4 flex flex-wrap gap-2">
          {patients.slice(0, 12).map((p) => (
            <button
              key={p.patient_id}
              type="button"
              onClick={() => setPatientId(p.patient_id)}
              className="rounded-full border border-slate-600 bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:border-cyan-300 hover:text-cyan-200"
            >
              {p.name} ({p.patient_id})
            </button>
          ))}
        </div>
      ) : null}

      {error ? <p className="text-sm text-rose-300">{error}</p> : null}

      <form onSubmit={onRagSubmit} className="glass mb-4 rounded-2xl p-4">
        <p className="text-sm font-semibold text-cyan-200">Smart RAG Summary (Grounded in Stored Patient Records)</p>
        <p className="mt-1 text-xs text-slate-300">
          Ask patient-history questions such as progression trend, prior findings, or scan-date-wise summary.
        </p>
        <div className="mt-2 flex flex-col gap-2 md:flex-row">
          <input
            value={ragQuery}
            onChange={(e) => setRagQuery(e.target.value)}
            className="flex-1 rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm"
          />
          <button
            disabled={ragLoading}
            className="rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {ragLoading ? "Generating..." : "Generate Summary"}
          </button>
        </div>
        {ragError ? <p className="mt-2 text-xs text-rose-300">{ragError}</p> : null}
        {ragAnswer ? <p className="mt-2 text-sm text-slate-100">{ragAnswer}</p> : null}
        {ragCitations.length ? (
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            {ragCitations.map((cite) => (
              <p key={cite}>{cite}</p>
            ))}
          </div>
        ) : null}
      </form>

      {profile ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="glass rounded-2xl p-4">
            <h2 className="text-lg font-semibold text-cyan-200">Patient Timeline</h2>
            <div className="mt-3 space-y-3">
              {profile.scans.map((row) => {
                const scan = row.scan;
                return (
                  <div key={scan.id} className="rounded-xl border border-slate-700 bg-slate-900/50 p-3 text-sm text-slate-200">
                    <p className="font-semibold">Scan #{scan.id} | {scan.scan_date}</p>
                    <p>Tumor: {scan.tumor_detected ? "Detected" : "Not detected"}</p>
                    <p>Type: {scan.tumor_type ?? "N/A"}</p>
                    <p>Stage: {scan.stage_label ?? "N/A"}</p>
                    <p>Risk: {scan.risk_category}</p>
                    <p>Confidence: {(scan.confidence_score * 100).toFixed(2)}%</p>
                    <div className="mt-2 grid grid-cols-2 gap-2">
                      {row.assets.image_url ? <img src={toAbsoluteUrl(row.assets.image_url) ?? ""} alt="MRI" className="h-20 w-full rounded object-cover" /> : null}
                      {row.assets.mask_url ? <img src={toAbsoluteUrl(row.assets.mask_url) ?? ""} alt="Mask" className="h-20 w-full rounded object-cover" /> : null}
                      {row.assets.corrected_mask_url ? <img src={toAbsoluteUrl(row.assets.corrected_mask_url) ?? ""} alt="Corrected Mask" className="h-20 w-full rounded object-cover" /> : null}
                      {row.assets.gradcam_url ? <img src={toAbsoluteUrl(row.assets.gradcam_url) ?? ""} alt="Grad-CAM" className="h-20 w-full rounded object-cover" /> : null}
                      {row.assets.overlay_url ? <img src={toAbsoluteUrl(row.assets.overlay_url) ?? ""} alt="Overlay" className="h-20 w-full rounded object-cover" /> : null}
                    </div>
                    {row.assets.report_url ? (
                      <a href={toAbsoluteUrl(row.assets.report_url) ?? undefined} target="_blank" rel="noreferrer" className="mt-2 inline-flex rounded-lg bg-cyan-400 px-3 py-1 text-xs font-semibold text-slate-950">
                        Open Report
                      </a>
                    ) : null}
                    <a
                      href={`/comparison?patient_id=${profile.patient.patient_id}&previous_scan_id=${scan.id}`}
                      className="ml-2 mt-2 inline-flex rounded-lg border border-slate-600 px-3 py-1 text-xs font-semibold text-slate-200 hover:border-cyan-300 hover:text-cyan-200"
                    >
                      Compare From This Scan
                    </a>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="glass rounded-2xl p-4">
            <h2 className="text-lg font-semibold text-cyan-200">Tumor Progression Graph</h2>
            {chartData ? <Line data={chartData} /> : <p className="mt-3 text-sm text-slate-300">No chart data yet.</p>}
            <p className="mt-3 text-xs text-slate-300">If scans are 2D, graph is area-based approximation (not true clinical volume).</p>
            <div className="mt-4">
              <VolumeSliceViewer title="Latest Scan 3D Slice Viewer" sliceUrls={latestVolumeSlices} selectedSliceIndex={latestSelectedSlice} />
            </div>
          </div>
        </div>
      ) : (
        <div className="glass rounded-2xl p-4 text-sm text-slate-300">No patient selected.</div>
      )}
    </PageShell>
  );
}
