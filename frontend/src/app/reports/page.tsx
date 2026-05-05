"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import PageShell from "@/components/PageShell";
import { api } from "@/lib/api";
import { toAbsoluteUrl } from "@/lib/assets";
import { ReportListItem } from "@/lib/types";

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [patientIdFilter, setPatientIdFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [cmpPatientId, setCmpPatientId] = useState("");
  const [previousScanId, setPreviousScanId] = useState("");
  const [currentScanId, setCurrentScanId] = useState("");
  const [cmpError, setCmpError] = useState<string | null>(null);
  const [cmpUrl, setCmpUrl] = useState<string | null>(null);

  async function fetchReports(patientId?: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get("/api/reports", {
        params: patientId ? { patient_id: patientId } : undefined
      });
      setReports(Array.isArray(res.data?.items) ? res.data.items : []);
    } catch {
      setError("Unable to fetch reports right now.");
      setReports([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchReports();
  }, []);

  async function onFilter(e: FormEvent) {
    e.preventDefault();
    await fetchReports(patientIdFilter.trim() || undefined);
  }

  const sortedReports = useMemo(
    () => [...reports].sort((a, b) => new Date(b.scan_date).getTime() - new Date(a.scan_date).getTime()),
    [reports]
  );

  function onBuildComparisonReport(e: FormEvent) {
    e.preventDefault();
    setCmpError(null);
    setCmpUrl(null);
    if (!cmpPatientId || !previousScanId || !currentScanId) {
      setCmpError("Provide patient ID, previous scan ID, and current scan ID.");
      return;
    }
    const url = `/api/reports/comparison/${encodeURIComponent(cmpPatientId)}/${encodeURIComponent(previousScanId)}/${encodeURIComponent(currentScanId)}`;
    setCmpUrl(toAbsoluteUrl(url));
  }

  return (
    <PageShell title="Report Download" subtitle="Download single-scan reports and generate longitudinal comparison reports for patient follow-up.">
      <div className="grid gap-4 lg:grid-cols-2">
        <form onSubmit={onFilter} className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Find Patient Reports</h2>
          <div className="mt-3 flex gap-2">
            <input
              value={patientIdFilter}
              onChange={(e) => setPatientIdFilter(e.target.value)}
              placeholder="Patient ID (optional)"
              className="w-full rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm"
            />
            <button className="rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300">Search</button>
          </div>
          {error ? <p className="mt-2 text-xs text-rose-300">{error}</p> : null}
        </form>

        <form onSubmit={onBuildComparisonReport} className="glass rounded-2xl p-4">
          <h2 className="text-lg font-semibold text-cyan-200">Generate Comparison Report PDF</h2>
          <div className="mt-3 grid gap-2 md:grid-cols-3">
            <input value={cmpPatientId} onChange={(e) => setCmpPatientId(e.target.value)} placeholder="Patient ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm" />
            <input value={previousScanId} onChange={(e) => setPreviousScanId(e.target.value)} placeholder="Previous Scan ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm" />
            <input value={currentScanId} onChange={(e) => setCurrentScanId(e.target.value)} placeholder="Current Scan ID" className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-sm" />
          </div>
          <button className="mt-3 rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300">Build Comparison PDF</button>
          {cmpError ? <p className="mt-2 text-xs text-rose-300">{cmpError}</p> : null}
          {cmpUrl ? (
            <a href={cmpUrl} target="_blank" rel="noreferrer" className="mt-3 inline-flex rounded-xl border border-cyan-300/40 bg-cyan-500/10 px-3 py-1.5 text-sm text-cyan-100">
              Download Comparison Report
            </a>
          ) : null}
        </form>
      </div>

      <div className="mt-4 glass rounded-2xl p-4">
        <h2 className="text-lg font-semibold text-cyan-200">Available Scan Reports</h2>
        {loading ? <p className="mt-2 text-sm text-slate-300">Loading report index...</p> : null}
        {!loading && !sortedReports.length ? <p className="mt-2 text-sm text-slate-300">No reports found.</p> : null}
        <div className="mt-3 space-y-2">
          {sortedReports.map((item) => (
            <div key={item.scan_id} className="rounded-xl border border-slate-700 bg-slate-900/50 p-3 text-sm">
              <p className="font-semibold text-slate-100">
                Scan #{item.scan_id} | {item.scan_date}
              </p>
              <p className="text-slate-300">
                Patient: {item.patient_name ?? "N/A"} ({item.patient_id ?? "N/A"})
              </p>
              <p className="text-slate-300">
                Tumor: {item.tumor_detected ? "Detected" : "Not detected"} | Type: {item.tumor_type ?? "N/A"} | Stage: {item.stage_label ?? "N/A"}
              </p>
              <a
                href={toAbsoluteUrl(item.report_url) ?? undefined}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex rounded-lg bg-cyan-400 px-3 py-1 text-xs font-semibold text-slate-950 hover:bg-cyan-300"
              >
                Download PDF
              </a>
            </div>
          ))}
        </div>
      </div>
    </PageShell>
  );
}
