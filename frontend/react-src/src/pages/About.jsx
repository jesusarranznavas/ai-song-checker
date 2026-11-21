import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Play } from 'lucide-react'
import useReveal from '../hooks/useReveal.js'

export default function About() {
  const navigate = useNavigate()
  const reveal = useReveal()

  return (
    <div className="container" ref={reveal}>
      <div className="about-hero reveal">
        <div className="eyebrow">Final Degree Project</div>
        <h1 className="about-title" style={{ marginTop: 18 }}>
          Detecting the <span className="text-grad">human signature</span> in audio.
        </h1>
      </div>

      <div className="about-grid section" style={{ paddingBottom: 0 }}>
        <div className="about-body reveal">
          <p>
            IA Analyzer is a <strong>Final Degree Project</strong> built around a single
            question: as generative models get better at producing music, can we still tell what a
            human made?
          </p>
          <p>
            The system processes audio <strong>entirely on the server</strong>. A track is
            normalized, divided into overlapping temporal windows, and each window is turned into a
            learned representation. A classification model then estimates three probabilities &mdash;{' '}
            <strong>human, AI-modified and pure AI</strong> &mdash; which are aggregated and presented as
            interpretable percentages.
          </p>
          <p>
            The goal is not a black box that says &ldquo;fake&rdquo; or &ldquo;real&rdquo;, but a{' '}
            <strong>transparent tool</strong> that surfaces confidence, entropy and per-window
            behaviour, so the result can be reasoned about rather than blindly trusted.
          </p>
          <button className="btn-solid cta" style={{ marginTop: 32 }} onClick={() => navigate('/analyzer')}>
            <Play size={18} fill="currentColor" stroke="none" /> Try the analyzer
          </button>
        </div>
      </div>
    </div>
  )
}
