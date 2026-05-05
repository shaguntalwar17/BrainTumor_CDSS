export default function AnimatedMedicalBackground() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="brain-orb brain-orb-a" />
      <div className="brain-orb brain-orb-b" />
      <div className="brain-orb brain-orb-c" />
      <div className="brain-grid" />
    </div>
  );
}
