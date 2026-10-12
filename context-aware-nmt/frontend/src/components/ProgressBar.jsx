export default function ProgressBar({ percent, label }) {
  return (
    <div>
      {label && (
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-600">{label}</span>
          <span className="font-medium text-slate-900">{percent}%</span>
        </div>
      )}
      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-brand-600 transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        ></div>
      </div>
    </div>
  )
}
