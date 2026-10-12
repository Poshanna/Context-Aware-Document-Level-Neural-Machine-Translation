export default function ComparePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 mb-2">Pipeline Comparison</h1>
        <p className="text-sm text-slate-600">
          Side-by-side sentence-level view: baseline translation vs. context-aware translation,
          with per-sentence deltas when a reference is provided.
        </p>
      </div>
      <div className="card">
        <div className="card-header">
          <h2 className="font-semibold text-slate-900">Select a translation job to compare</h2>
        </div>
        <div className="card-body text-sm text-slate-500 text-center py-12">
          Upload a document with a reference translation, or pick a past job from History, to
          enable per-sentence delta comparison here.
        </div>
      </div>
    </div>
  )
}
