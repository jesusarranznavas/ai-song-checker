import React, { useMemo, useRef, useState } from 'react'
import { Upload, Trash2 } from 'lucide-react'
import { analyzeFile } from '../api.js'
import useReveal from '../hooks/useReveal.js'

const toPercent = (v) => `${(v * 100).toFixed(1)}%`
const clamp01 = (v) => Math.max(0, Math.min(1, v))

function formatBytes(b) {
  if (b < 1024) return `${b} B`
  const kb = b / 1024; if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024; if (mb < 1024) return `${mb.toFixed(2)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}
function formatDate(d) { try { return new Date(d).toLocaleString() } catch { return d } }

function MetricBar({ label, value, cls }) {
  const v = clamp01(value)
  return (
    <div className="metric">
      <div className="metric-top">
        <span className="metric-label">{label}</span>
        <span className="metric-val">{toPercent(v)}</span>
      </div>
      <div className="metric-track">
        <div className={`metric-fill ${cls}`} style={{ width: `${v * 100}%` }} />
      </div>
    </div>
  )
}

export default function Analyzer() {
  const inputRef = useRef(null)
  const reveal = useReveal()

  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const history = useMemo(() => {
    try { return JSON.parse(localStorage.getItem('history') || '[]') } catch { return [] }
  }, [result])

  function saveHistory(a) {
    let cur = []
    try { cur = JSON.parse(localStorage.getItem('history') || '[]') } catch { /* noop */ }
    cur.unshift({ at: new Date().toISOString(), filename: a.filename, human: a.human, ai_modified: a.ai_modified, pure_ai: a.pure_ai, inconclusive: a.inconclusive })
    localStorage.setItem('history', JSON.stringify(cur.slice(0, 20)))
  }
  function clearHistory() { localStorage.removeItem('history'); setResult((p) => (p ? { ...p } : null)) }

  function pickFile(f) { setError(null); setResult(null); setProgress(0); setFile(f || null) }

  async function analyze() {
    if (!file) return
    setLoading(true); setError(null); setResult(null); setProgress(8)
    const timer = window.setInterval(() => {
      setProgress((p) => (p < 55 ? p + 3 : p < 78 ? p + 1.4 : p < 92 ? p + 0.6 : p))
    }, 220)
    try {
      const analysis = await analyzeFile(file)
      window.clearInterval(timer)
      setProgress(100)
      setTimeout(() => { setResult(analysis); saveHistory(analysis); setLoading(false) }, 250)
    } catch (err) {
      window.clearInterval(timer)
      setError(err?.message || 'Unexpected error'); setLoading(false); setProgress(0)
    }
  }

  function onDrop(e) {
    e.preventDefault(); e.stopPropagation(); setDragActive(false)
    const f = e.dataTransfer.files?.[0]; if (f) pickFile(f)
  }

  const badge = result
    ? (result.inconclusive
        ? <span className="badge warning">Review suggested</span>
        : <span className="badge success">Confident</span>)
    : null

  return (
    <div className="container" ref={reveal}>
      <div className="page-head reveal">
        <div className="eyebrow">Analyzer</div>
        <h1 className="display" style={{ fontSize: 'clamp(2.4rem,5vw,4rem)', marginTop: 18 }}>AI Audio Analyzer</h1>
        <p className="lead" style={{ marginTop: 18, maxWidth: '52ch' }}>
          Upload a track and review the estimated distribution returned by the model across
          sliding time windows.
        </p>
      </div>

      <div className="analyzer-grid">
        {/* UPLOAD */}
        <section className="panel liquid-glass reveal">
          <div className="panel-head">
            <h2>Upload audio</h2>
            <p>Select a file or drag it into the area below.</p>
          </div>

          <div
            className={'dropzone' + (dragActive ? ' active' : '')}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            onClick={(e) => { if (e.target.closest('button')) return; inputRef.current?.click() }}
          >
            <input ref={inputRef} type="file" accept="audio/*" style={{ display: 'none' }}
              onChange={(e) => pickFile(e.target.files?.[0] || null)} />

            {!file && (
              <div>
                <div className="dz-ico liquid-glass"><Upload size={26} /></div>
                <div className="dz-title">Drop audio file here</div>
                <div className="dz-text">MP3 · WAV · FLAC · OGG</div>
                <button className="btn-solid dz-btn" onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}>Select file</button>
              </div>
            )}

            {file && (
              <div className="file-card">
                <div>
                  <div className="file-name">{file.name}</div>
                  <div className="file-meta">
                    <span>{formatBytes(file.size)}</span><span>·</span><span>{file.type || 'audio file'}</span>
                  </div>
                </div>
                <div className="file-actions">
                  <button className="btn-solid" disabled={loading} onClick={(e) => { e.stopPropagation(); analyze() }}>
                    {loading ? 'Analyzing…' : 'Start analysis'}
                  </button>
                  <button className="liquid-glass btn-glass" disabled={loading} onClick={(e) => { e.stopPropagation(); pickFile(null) }}>Remove</button>
                </div>
                {(loading || progress > 0) && (
                  <div className="progress">
                    <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
                    <div className="progress-val">{Math.round(progress)}%</div>
                  </div>
                )}
              </div>
            )}
          </div>

          {error && (
            <div className="msg error" style={{ marginTop: 16 }}>
              <strong>Request failed.</strong>
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* RESULT */}
        <section className="panel liquid-glass reveal">
          <div className="panel-head">
            <h2>Analysis result</h2>
          </div>

          {!result && (
            <div className="empty">
              <div className="empty-title">No analysis yet</div>
              <div className="empty-text">Upload a file and run the analysis to view the result.</div>
            </div>
          )}

          {result && (
            <div className="result">
              <div className="result-top">
                <div>
                  <div className="result-file">{result.filename}</div>
                  <div className="result-meta">
                    <span>{result.duration_sec.toFixed(1)} s</span>
                  </div>
                </div>
                <div>{badge}</div>
              </div>

              <div className="metrics">
                <MetricBar label="Human" value={result.human} cls="human" />
                <MetricBar label="AI-modified" value={result.ai_modified} cls="mod" />
                <MetricBar label="Pure AI" value={result.pure_ai} cls="ai" />
              </div>

              {result.inconclusive && (
                <div className="msg warning">
                  <strong>Manual review suggested.</strong>
                  <span>The signal appears intermediate or low-confidence. Common with strongly processed or hybrid material.</span>
                </div>
              )}
            </div>
          )}
        </section>

        {/* HISTORY */}
        <section className="panel panel-full liquid-glass reveal">
          <div className="panel-head panel-head-row">
            <div>
              <h2>History</h2>
              <p>Recent analyses stored locally in this browser.</p>
            </div>
            <button className="liquid-glass btn-glass" style={{ padding: '10px 18px', fontSize: '0.88rem', display: 'inline-flex', alignItems: 'center', gap: 8 }} onClick={clearHistory} disabled={history.length === 0}>
              <Trash2 size={15} /> Clear
            </button>
          </div>

          {history.length === 0 && <div className="history-empty">No saved analyses yet.</div>}

          {history.length > 0 && (
            <div className="table-wrap">
              <table className="h-table">
                <thead>
                  <tr>{['File', 'Human', 'AI-modified', 'Pure AI', 'Status', 'Date'].map((h) => <th key={h}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {history.map((it, i) => (
                    <tr key={`${it.filename}-${i}`}>
                      <td>{it.filename}</td>
                      <td>{toPercent(it.human)}</td>
                      <td>{toPercent(it.ai_modified)}</td>
                      <td>{toPercent(it.pure_ai)}</td>
                      <td><span className={'badge ' + (it.inconclusive ? 'warning' : 'success')}>{it.inconclusive ? 'Review' : 'OK'}</span></td>
                      <td>{formatDate(it.at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
