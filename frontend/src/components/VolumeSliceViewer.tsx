"use client";

import { useEffect, useMemo, useState } from "react";

import { toAbsoluteUrl } from "@/lib/assets";

type Props = {
  title?: string;
  sliceUrls: string[];
  selectedSliceIndex?: number | null;
};

export default function VolumeSliceViewer({ title = "3D Slice Viewer", sliceUrls, selectedSliceIndex }: Props) {
  const [index, setIndex] = useState(() => {
    if (!sliceUrls.length) return 0;
    if (selectedSliceIndex === null || selectedSliceIndex === undefined) return Math.floor(sliceUrls.length / 2);
    return Math.max(0, Math.min(sliceUrls.length - 1, selectedSliceIndex));
  });

  const currentUrl = useMemo(() => {
    if (!sliceUrls.length) return null;
    return toAbsoluteUrl(sliceUrls[index] ?? null);
  }, [sliceUrls, index]);

  useEffect(() => {
    if (!sliceUrls.length) {
      setIndex(0);
      return;
    }
    if (selectedSliceIndex === null || selectedSliceIndex === undefined) {
      setIndex(Math.floor(sliceUrls.length / 2));
      return;
    }
    setIndex(Math.max(0, Math.min(sliceUrls.length - 1, selectedSliceIndex)));
  }, [sliceUrls, selectedSliceIndex]);

  if (!sliceUrls.length) {
    return (
      <div className="glass rounded-2xl p-4">
        <p className="text-sm font-semibold text-cyan-200">{title}</p>
        <p className="mt-2 text-sm text-slate-300">No 3D slice stack available for this scan.</p>
      </div>
    );
  }

  return (
    <div className="glass rounded-2xl p-4">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-sm font-semibold text-cyan-200">{title}</p>
        <p className="text-xs text-slate-300">
          Slice {index + 1} / {sliceUrls.length}
        </p>
      </div>
      {currentUrl ? <img src={currentUrl} alt={`Volume slice ${index + 1}`} className="h-56 w-full rounded-lg object-contain bg-slate-950/50" /> : null}
      <input
        type="range"
        min={0}
        max={Math.max(0, sliceUrls.length - 1)}
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="mt-3 w-full accent-cyan-400"
      />
      <p className="mt-2 text-xs text-slate-300">
        Interactive slice-scrolling view for volumetric MRI uploads. Window/level and PACS-level tools are future extensions.
      </p>
    </div>
  );
}
