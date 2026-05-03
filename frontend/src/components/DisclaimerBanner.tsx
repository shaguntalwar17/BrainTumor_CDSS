export default function DisclaimerBanner() {
  return (
    <div className="glass rounded-xl border-l-4 border-amber-300 px-4 py-3 text-sm text-slate-100">
      <p className="font-semibold">Important Medical Disclaimer</p>
      <p>
        This system is an AI-assisted academic prototype for Brain MRI tumor analysis. It is not a certified medical diagnostic tool.
        All predictions, segmentations, heatmaps, and reports must be verified by a qualified radiologist or medical professional.
      </p>
      <p className="mt-2 text-amber-200">
        Tumor stage prediction is not provided because the dataset does not include clinically validated staging labels.
      </p>
    </div>
  );
}
