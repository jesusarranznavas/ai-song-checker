import React from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Play, ArrowRight, BarChart3, Upload, Waves, Cpu,
} from 'lucide-react'
import useReveal from '../hooks/useReveal.js'

function MiniWave() {
  const bars = [0.3, 0.7, 0.45, 0.9, 0.55, 0.8, 0.4, 0.95, 0.6, 0.75, 0.5, 0.85, 0.35, 0.7]
  return (
    <div className="mini-wave" aria-hidden="true">
      {bars.map((h, i) => (
        <span key={i} style={{ height: `${h * 100}%`, animationDelay: `${i * 0.08}s` }} />
      ))}
    </div>
  )
}

function PreviewCard() {
  return (
    <div className="preview-card liquid-glass animate-blur-fade-up" style={{ animationDelay: '700ms' }}>
      <div className="preview-head">
        <span className="preview-dot" />
        <span>Live sample · analyzing</span>
      </div>
      <MiniWave />
      <div className="preview-metrics">
        <div className="pm"><span className="pm-lbl">Human</span><span className="pm-val">62.4%</span></div>
        <div className="pm"><span className="pm-lbl">AI-mod</span><span className="pm-val">24.1%</span></div>
        <div className="pm"><span className="pm-lbl">Pure AI</span><span className="pm-val">13.5%</span></div>
      </div>
    </div>
  )
}

const STEPS = [
  { n: '01', Icon: Upload, title: 'Upload', body: 'Drop or pick an audio file. It is sent to the server for processing — no data is stored.' },
  { n: '02', Icon: Waves, title: 'Windowing', body: 'The signal is normalized and split into overlapping temporal windows for analysis.' },
  { n: '03', Icon: Cpu, title: 'Inference', body: 'Each window becomes a learned embedding and is classified by the trained model.' },
  { n: '04', Icon: BarChart3, title: 'Result', body: 'Output is aggregated into interpretable human, AI-modified and pure-AI percentages.' },
]

export default function Home() {
  const navigate = useNavigate()
  const reveal = useReveal()

  return (
    <div ref={reveal}>
      {/* HERO */}
      <section className="hero container">
        <div className="hero-row">
          <div className="hero-left">
            <h1 className="display hero-title animate-blur-fade-up" style={{ animationDelay: '400ms' }}>
              Where <span className="text-grad">human</span> ends<br />and <span className="text-grad">AI</span> begins.
            </h1>
            <p className="lead hero-desc animate-blur-fade-up" style={{ animationDelay: '500ms' }}>
              A final degree project that listens to a track and estimates how much of it is
              human-made, AI-modified, or purely generated — with results you can actually read.
            </p>
            <div className="cta-row">
              <button className="btn-solid cta animate-blur-fade-up" style={{ animationDelay: '600ms' }} onClick={() => navigate('/analyzer')}>
                <Play size={18} fill="currentColor" stroke="none" /> Open Analyzer
              </button>
              <button className="liquid-glass btn-glass cta animate-blur-fade-up" style={{ animationDelay: '700ms' }} onClick={() => navigate('/about')}>
                About the project <ArrowRight size={18} />
              </button>
            </div>
          </div>
          <div className="hero-right"><PreviewCard /></div>
        </div>
        <div className="scroll-cue">
          <span className="line" />
          <span>Scroll</span>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="section container">
        <div className="section-head reveal">
          <div className="eyebrow">Process</div>
          <h2>Four steps, fully server-side.</h2>
        </div>
        <div className="steps">
          {STEPS.map((s) => (
            <article key={s.n} className="step liquid-glass reveal">
              <div className="step-ico liquid-glass"><s.Icon size={22} /></div>
              <div className="step-num">{s.n}</div>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </article>
          ))}
        </div>
      </section>

      {/* STAT STRIP */}
      <section className="section container" style={{ paddingTop: 0 }}>
        <div className="stat-strip liquid-glass reveal">
          <div className="stat"><div className="num text-grad">3</div><div className="lbl">Classes: human · modified · AI</div></div>
          <div className="stat"><div className="num text-grad">100%</div><div className="lbl">Local processing</div></div>
          <div className="stat"><div className="num text-grad">0</div><div className="lbl">Data retained on server</div></div>
        </div>
      </section>

      {/* CTA BAND */}
      <section className="section container" style={{ paddingTop: 0 }}>
        <div className="cta-band liquid-glass reveal">
          <div>
            <h2 style={{ fontSize: 'clamp(1.8rem,3.4vw,2.8rem)', letterSpacing: '-0.04em' }}>Ready to test a track?</h2>
            <p className="lead" style={{ marginTop: 12 }}>Drop a song and see the breakdown in seconds.</p>
          </div>
          <button className="btn-solid cta" onClick={() => navigate('/analyzer')}>
            <Play size={18} fill="currentColor" stroke="none" /> Open Analyzer
          </button>
        </div>
      </section>
    </div>
  )
}
