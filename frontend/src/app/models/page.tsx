"use client";

import { useEffect, useState } from "react";

import PageShell from "@/components/PageShell";
import { api } from "@/lib/api";
import { ModelMetric } from "@/lib/types";

function fmt(v: number | null | undefined, digits = 4) {
  if (v === null || v === undefined) return "N/A";
  return Number(v).toFixed(digits);
}

export default function ModelsPage() {
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [note, setNote] = useState("");

  useEffect(() => {
    api
      .get("/model-metrics")
      .then((res) => {
        setMetrics(res.data.items ?? []);
        setNote(res.data.note ?? "");
      })
      .catch(() => {
        setMetrics([]);
        setNote("Unable to fetch metrics.");
      });
  }, []);

  return (
    <PageShell title="Model Performance Leaderboard" subtitle="Compare classification and segmentation models using actual run metrics. Status shows real/demo baseline readiness.">
      <div className="glass overflow-x-auto rounded-2xl p-4">
        <table className="min-w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-600 text-cyan-200">
              <th className="px-2 py-2">Model</th>
              <th className="px-2 py-2">Task</th>
              <th className="px-2 py-2">Accuracy</th>
              <th className="px-2 py-2">F1</th>
              <th className="px-2 py-2">AUC</th>
              <th className="px-2 py-2">Dice</th>
              <th className="px-2 py-2">IoU</th>
              <th className="px-2 py-2">Infer Time</th>
              <th className="px-2 py-2">Size (MB)</th>
              <th className="px-2 py-2">Status</th>
              <th className="px-2 py-2">Best Use Case</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m.id} className="border-b border-slate-800 text-slate-200">
                <td className="px-2 py-2 font-semibold">{m.model_name}</td>
                <td className="px-2 py-2">{m.task_type}</td>
                <td className="px-2 py-2">{fmt(m.accuracy)}</td>
                <td className="px-2 py-2">{fmt(m.f1_score)}</td>
                <td className="px-2 py-2">{fmt(m.auc)}</td>
                <td className="px-2 py-2">{fmt(m.dice)}</td>
                <td className="px-2 py-2">{fmt(m.iou)}</td>
                <td className="px-2 py-2">{fmt(m.inference_time)}</td>
                <td className="px-2 py-2">{fmt(m.model_size, 2)}</td>
                <td className="px-2 py-2">{m.status ?? "demo"}</td>
                <td className="px-2 py-2 text-xs text-slate-300">{m.best_use_case ?? "N/A"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-slate-300">{note}</p>
    </PageShell>
  );
}
