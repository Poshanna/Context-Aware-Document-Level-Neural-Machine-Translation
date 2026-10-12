export default function PipelineSelector({ value, onChange }) {
  const options = [
    { code: 'baseline', label: 'Baseline (sentence-level)', desc: 'No context — each sentence translated independently' },
    { code: 'context_aware', label: 'Context-aware (document-level)', desc: 'Prior-sentence context retrieved and injected' },
  ]

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">
        Pipeline
      </label>
      <div className="space-y-2">
        {options.map(o => (
          <label
            key={o.code}
            className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
              value === o.code
                ? 'border-brand-500 bg-brand-50'
                : 'border-slate-200 hover:bg-slate-50'
            }`}
          >
            <input
              type="radio"
              value={o.code}
              checked={value === o.code}
              onChange={() => onChange(o.code)}
              className="mt-1"
            />
            <div>
              <div className="font-medium text-slate-900">{o.label}</div>
              <div className="text-xs text-slate-500 mt-0.5">{o.desc}</div>
            </div>
          </label>
        ))}
      </div>
    </div>
  )
}
