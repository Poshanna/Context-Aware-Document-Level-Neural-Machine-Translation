export default function LanguageSelector({ value, onChange, label, exclude }) {
  const options = [
    { code: 'te', label: 'Telugu' },
    { code: 'en', label: 'English' },
    { code: 'hi', label: 'Hindi' },
  ].filter(o => o.code !== exclude)

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">
        {label}
      </label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="input"
      >
        {options.map(o => (
          <option key={o.code} value={o.code}>
            {o.label} ({o.code})
          </option>
        ))}
      </select>
    </div>
  )
}
