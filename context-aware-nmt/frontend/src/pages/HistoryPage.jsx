export default function HistoryPage() {
  const sample = [
    { id: 'a', pipeline: 'baseline', model_name: 'nllb200', source_lang: 'te', target_lang: 'en', status: 'completed', created_at: '2026-09-29T10:00:00Z', source_text_preview: 'మీరు ఎలా ఉన్నారు?' },
    { id: 'b', pipeline: 'context_aware', model_name: 'indictrans2', source_lang: 'en', target_lang: 'te', status: 'completed', created_at: '2026-09-28T15:30:00Z', source_text_preview: 'The quick brown fox...' },
  ]
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900">Translation History</h1>
        <div className="text-sm text-slate-500">2 items (sign in to see yours)</div>
      </div>
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              {['Time', 'Direction', 'Pipeline', 'Model', 'Status', 'Preview', 'Actions'].map(h => (
                <th key={h} className="text-left px-4 py-3 font-medium text-slate-600">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sample.map(r => (
              <tr key={r.id} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="px-4 py-3 text-slate-500 font-mono text-xs">
                  {new Date(r.created_at).toLocaleString()}
                </td>
                <td className="px-4 py-3 font-medium">{r.source_lang} → {r.target_lang}</td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    r.pipeline === 'context_aware' ? 'bg-brand-50 text-brand-700' : 'bg-slate-100 text-slate-700'
                  }`}>
                    {r.pipeline}
                  </span>
                </td>
                <td className="px-4 py-3">{r.model_name}</td>
                <td className="px-4 py-3 text-green-700">{r.status}</td>
                <td className="px-4 py-3 max-w-xs truncate text-slate-600">{r.source_text_preview}</td>
                <td className="px-4 py-3 space-x-2">
                  <button className="text-xs text-brand-700 hover:underline">View</button>
                  <button className="text-xs text-slate-600 hover:underline">Replay</button>
                  <button className="text-xs text-red-600 hover:underline">Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
