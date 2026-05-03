type MetricCardProps = {
  title: string;
  value: string;
  subtitle?: string;
  tone?: "cyan" | "emerald" | "amber" | "violet";
};

const toneMap = {
  cyan: "from-cyan-500/25 to-sky-500/10 border-cyan-300/30",
  emerald: "from-emerald-500/25 to-teal-500/10 border-emerald-300/30",
  amber: "from-amber-500/25 to-yellow-500/10 border-amber-300/30",
  violet: "from-indigo-500/25 to-violet-500/10 border-violet-300/30"
};

export default function MetricCard({ title, value, subtitle, tone = "cyan" }: MetricCardProps) {
  return (
    <div className={`glass rounded-2xl border bg-gradient-to-br p-4 ${toneMap[tone]}`}>
      <p className="text-xs uppercase tracking-widest text-slate-300">{title}</p>
      <p className="mt-2 text-2xl font-bold text-white">{value}</p>
      {subtitle ? <p className="mt-2 text-xs text-slate-300">{subtitle}</p> : null}
    </div>
  );
}
