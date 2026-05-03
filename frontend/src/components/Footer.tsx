export default function Footer() {
  return (
    <footer className="mt-10 border-t border-slate-700/70 bg-slate-950/70 px-4 py-6 text-sm text-slate-300 md:px-6">
      <div className="mx-auto flex max-w-7xl flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <p>
          AI-assisted educational/research prototype. Not a clinical diagnostic tool. Verify with a certified radiologist/doctor.
        </p>
        <p className="font-semibold text-cyan-300">Project made by Shagun Talwar</p>
      </div>
    </footer>
  );
}
