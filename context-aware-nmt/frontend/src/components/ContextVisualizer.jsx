export default function ContextVisualizer({ sentences = [] }) {
  return (
    <div className="space-y-3">
      {sentences.map((s, i) => (
        <div key={i} className="card-body p-4 rounded-lg border border-slate-200">
          <div className="flex items-start justify-between gap-4">
            <div className="text-xs font-mono text-slate-400 mt-0.5">#{i + 1}</div>
            <div className="flex-1 space-y-2">
              <div className="text-sm text-slate-700 bg-slate-50 p-2 rounded">
                Source: {s.source || '—'}
              </div>
              <div className="text-sm text-slate-900 p-2 rounded bg-brand-50 border border-brand-100">
                Target: {s.target || '—'}
              </div>
              {s.context && s.context.length > 0 && (
                <div className="mt-3 border-t pt-2">
                  <div className="text-xs font-semibold text-slate-500 mb-2">
                    Retrieved context ({s.context.length})
                  </div>
                  <ul className="space-y-1">
                    {s.context.map((c, j) => (
                      <li
                        key={j}
                        className="text-xs bg-amber-50 border border-amber-200 p-2 rounded flex items-start gap-2"
                      >
                        <span className="font-mono text-amber-700 shrink-0">
                          S{c.source_sent_idx + 1} · {(c.score * 100).toFixed(0)}%
                        </span>
                        <span className="text-amber-900 line-clamp-2">{c.snippet}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
      {sentences.length === 0 && (
        <div className="text-sm text-slate-500 text-center py-8">
          Run a context-aware translation to see per-sentence context matches here.
        </div>
      )}
    </div>
  )
}
