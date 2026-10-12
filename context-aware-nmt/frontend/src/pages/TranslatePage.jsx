import { useState } from 'react'
import LanguageSelector from '../components/LanguageSelector.jsx'
import PipelineSelector from '../components/PipelineSelector.jsx'
import ModelSelector from '../components/ModelSelector.jsx'
import Spinner from '../components/Spinner.jsx'
import Alert from '../components/Alert.jsx'
import ContextVisualizer from '../components/ContextVisualizer.jsx'

export default function TranslatePage() {
  const [sourceLang, setSourceLang] = useState('te')
  const [targetLang, setTargetLang] = useState('en')
  const [pipeline, setPipeline] = useState('baseline')
  const [model, setModel] = useState('nllb200')
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleTranslate = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // TODO: Replace with actual API call via services/api.js
      await new Promise(r => setTimeout(r, 800))
      setResult({
        translated_text: '[Translation output goes here]',
        tokens_used: null,
        sentences: [],
      })
    } catch (err) {
      setError(err?.message || 'Translation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid lg:grid-cols-5 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Translation Settings</h2>
          </div>
          <div className="card-body space-y-5">
            <div className="grid grid-cols-2 gap-4">
              <LanguageSelector
                label="Source language"
                value={sourceLang}
                onChange={v => {
                  setSourceLang(v)
                  if (v === targetLang) setTargetLang(v === 'te' ? 'en' : 'te')
                }}
              />
              <LanguageSelector
                label="Target language"
                value={targetLang}
                onChange={setTargetLang}
                exclude={sourceLang}
              />
            </div>
            <ModelSelector value={model} onChange={setModel} />
            <PipelineSelector value={pipeline} onChange={setPipeline} />
          </div>
        </div>
      </div>

      <div className="lg:col-span-3 space-y-6">
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="font-semibold text-slate-900">Input Text</h2>
            <span className="text-xs text-slate-500">{text.length}/50,000</span>
          </div>
          <div className="card-body">
            <textarea
              value={text}
              onChange={e => setText(e.target.value.slice(0, 50_000))}
              placeholder="Paste Telugu or English text here..."
              rows={8}
              className="input resize-y"
            />
          </div>
          <div className="card-footer flex items-center justify-end gap-3">
            <button
              className="btn-secondary"
              onClick={() => {
                setText('')
                setResult(null)
                setError(null)
              }}
              disabled={loading}
            >
              Clear
            </button>
            <button
              className="btn-primary"
              onClick={handleTranslate}
              disabled={loading || !text.trim()}
            >
              {loading ? <Spinner size="sm" label="Translating..." /> : 'Translate'}
            </button>
          </div>
        </div>

        {error && (
          <Alert type="error">
            <strong>Error: </strong>
            {error}
          </Alert>
        )}

        {result && (
          <div className="card">
            <div className="card-header">
              <h2 className="font-semibold text-slate-900">Translation Output</h2>
            </div>
            <div className="card-body">
              <div className="p-4 rounded-lg bg-brand-50 border border-brand-100 text-slate-900 min-h-[6rem] whitespace-pre-wrap">
                {result.translated_text}
              </div>
              {result.tokens_used && (
                <div className="text-xs text-slate-500 mt-2">
                  Tokens: {result.tokens_used}
                </div>
              )}
            </div>
            {pipeline === 'context_aware' && (
              <div className="border-t border-slate-200">
                <div className="card-header">
                  <h3 className="font-semibold text-slate-800 text-sm">Context Influence</h3>
                </div>
                <div className="card-body">
                  <ContextVisualizer sentences={result.sentences || []} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
