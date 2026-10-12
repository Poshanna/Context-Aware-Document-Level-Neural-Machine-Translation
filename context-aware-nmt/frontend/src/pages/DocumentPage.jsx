import { useState } from 'react'
import LanguageSelector from '../components/LanguageSelector.jsx'
import PipelineSelector from '../components/PipelineSelector.jsx'
import ModelSelector from '../components/ModelSelector.jsx'
import Spinner from '../components/Spinner.jsx'
import Alert from '../components/Alert.jsx'
import ProgressBar from '../components/ProgressBar.jsx'

export default function DocumentPage() {
  const [sourceLang, setSourceLang] = useState('te')
  const [targetLang, setTargetLang] = useState('en')
  const [pipeline, setPipeline] = useState('baseline')
  const [model, setModel] = useState('nllb200')
  const [file, setFile] = useState(null)
  const [job, setJob] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      // TODO: Replace with actual API call
      await new Promise(r => setTimeout(r, 1200))
      setJob({
        id: crypto.randomUUID(),
        status: 'running',
        progress_percent: 42,
        total_sentences: 17,
      })
    } catch (err) {
      setError(err?.message || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid lg:grid-cols-5 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Document Settings</h2>
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
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Upload Document</h2>
          </div>
          <div className="card-body">
            <label className="block">
              <div className="border-2 border-dashed border-slate-300 rounded-xl p-10 text-center hover:border-brand-400 transition-colors cursor-pointer">
                <div className="text-5xl mb-3">📄</div>
                <div className="font-medium text-slate-900 mb-1">
                  {file ? file.name : 'Click or drag to upload'}
                </div>
                <div className="text-xs text-slate-500">
                  PDF · DOCX · TXT — Max 10 MB
                </div>
                <input
                  type="file"
                  accept=".txt,.pdf,.docx"
                  className="hidden"
                  onChange={e => setFile(e.target.files?.[0] || null)}
                />
              </div>
            </label>
            {file && (
              <div className="mt-3 text-xs text-slate-500 flex justify-between">
                <span>{file.name}</span>
                <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
          </div>
          <div className="card-footer flex justify-end">
            <button
              className="btn-primary"
              onClick={handleUpload}
              disabled={loading || !file}
            >
              {loading ? <Spinner size="sm" label="Uploading..." /> : 'Start Translation Job'}
            </button>
          </div>
        </div>

        {error && (
          <Alert type="error">
            <strong>Error: </strong>
            {error}
          </Alert>
        )}

        {job && (
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h2 className="font-semibold text-slate-900">Job Status</h2>
              <span className={`text-xs px-2 py-1 rounded-full ${
                job.status === 'completed' ? 'bg-green-100 text-green-700'
                : job.status === 'failed' ? 'bg-red-100 text-red-700'
                : 'bg-amber-100 text-amber-700'
              }`}>
                {job.status.toUpperCase()}
              </span>
            </div>
            <div className="card-body space-y-4">
              <div className="text-xs text-slate-500 font-mono">
                Job ID: {job.id}
              </div>
              <ProgressBar
                percent={job.progress_percent}
                label={`Sentences: ${Math.ceil(job.total_sentences * job.progress_percent / 100)} / ${job.total_sentences}`}
              />
              {job.status === 'completed' && (
                <div className="flex gap-2 pt-2">
                  <button className="btn-primary">Download TXT</button>
                  <button className="btn-secondary">Download DOCX</button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
