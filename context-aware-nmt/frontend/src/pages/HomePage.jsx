import { Link } from 'react-router-dom'

export default function HomePage() {
  const features = [
    {
      title: 'Text Translation',
      desc: 'Translate Telugu ↔ English text with baseline or context-aware pipelines, pluggable model backends.',
      href: '/translate',
      cta: 'Try it →',
    },
    {
      title: 'Document Translation',
      desc: 'Upload PDF, DOCX, or TXT files. Per-sentence translation preserves structure and page metadata.',
      href: '/document',
      cta: 'Upload file →',
    },
    {
      title: 'Evaluation Dashboard',
      desc: 'Side-by-side BLEU, chrF++, BERTScore (and optional COMET) for honest baseline vs. context-aware comparison.',
      href: '/evaluate',
      cta: 'Run evaluation →',
    },
    {
      title: 'Context Visualization',
      desc: 'See exactly which preceding sentences influenced each target sentence, with similarity scores.',
      href: '/compare',
      cta: 'Compare pipelines →',
    },
  ]

  return (
    <div className="space-y-16">
      <section className="text-center py-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-50 text-brand-700 text-xs font-medium mb-6">
          B.E. AI &amp; ML Final-Year Research Project
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold text-slate-900 tracking-tight mb-6">
          Context-Aware Document-Level
          <br />
          Neural Machine Translation
        </h1>
        <p className="max-w-2xl mx-auto text-lg text-slate-600 mb-8">
          Translate Telugu ↔ English documents while leveraging discourse context to improve
          co-reference, terminology consistency, and accuracy — with honest side-by-side
          evaluation against a sentence-level baseline.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link to="/translate" className="btn-primary text-base px-6 py-3">
            Start Translating
          </Link>
          <Link to="/evaluate" className="btn-secondary text-base px-6 py-3">
            View Evaluation
          </Link>
        </div>
        <div className="mt-6 text-xs text-slate-500">
          All models are pretrained checkpoints. No fabrication. See{' '}
          <code className="bg-slate-100 px-1.5 py-0.5 rounded">project_rules.md</code>
        </div>
      </section>

      <section className="grid sm:grid-cols-2 gap-6">
        {features.map(f => (
          <div key={f.title} className="card card-body">
            <h3 className="font-semibold text-lg text-slate-900 mb-2">{f.title}</h3>
            <p className="text-sm text-slate-600 mb-4">{f.desc}</p>
            <Link to={f.href} className="text-sm font-medium text-brand-700 hover:text-brand-800">
              {f.cta}
            </Link>
          </div>
        ))}
      </section>

      <section className="card card-body">
        <h2 className="font-semibold text-xl text-slate-900 mb-4">Research Pipeline</h2>
        <ol className="grid md:grid-cols-4 gap-4 text-sm">
          {[
            ['1', 'Input', 'Text or PDF/DOCX/TXT document'],
            ['2', 'Preprocess', 'Split sentences, extract metadata'],
            ['3', 'Retrieve + Translate', 'SBERT k-NN context + IndicTrans2 / NLLB-200'],
            ['4', 'Evaluate', 'BLEU · chrF++ · BERTScore · COMET'],
          ].map(([n, t, d]) => (
            <li key={n} className="p-4 rounded-lg bg-slate-50 border border-slate-100">
              <div className="text-2xl font-bold text-brand-600 mb-1">{n}</div>
              <div className="font-semibold text-slate-900">{t}</div>
              <div className="text-slate-500 mt-1">{d}</div>
            </li>
          ))}
        </ol>
      </section>
    </div>
  )
}
