export default function ModelSelector({ value, onChange }) {
  const options = [
    { code: 'nllb200', label: 'NLLB-200 Distilled 600M', desc: 'Meta — lighter, works on CPU in demo mode' },
    { code: 'indictrans2', label: 'IndicTrans2 1B', desc: 'AI4Bharat — higher Telugu quality, GPU recommended' },
  ]

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">
        Model checkpoint
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
