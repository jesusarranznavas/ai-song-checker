import React, { useEffect, useRef, useState } from 'react'

function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const rect = canvas.getBoundingClientRect()
  canvas.width = Math.max(1, Math.floor(rect.width * dpr))
  canvas.height = Math.max(1, Math.floor(rect.height * dpr))
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { ctx, w: rect.width, h: rect.height }
}

/**
 * Full-screen opening animation: an oscilloscope waveform + ripples, then the
 * "IA ANALYZER" title blur-fades in. Auto-dismisses (or on click).
 */
export default function Intro({ onDone }) {
  const ref = useRef(null)
  const [done, setDone] = useState(false)
  const [showText, setShowText] = useState(false)
  const doneRef = useRef(false)

  function finish() {
    if (doneRef.current) return
    doneRef.current = true
    setDone(true)
    setTimeout(() => onDone && onDone(), 1000)
  }

  useEffect(() => {
    const canvas = ref.current
    let raf, w, h, ctx
    let t = 0
    const start = performance.now()
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    function resize() {
      const s = setupCanvas(canvas)
      ctx = s.ctx; w = s.w; h = s.h
    }
    resize()
    window.addEventListener('resize', resize)

    const ripples = []
    let nextRipple = 0

    function draw(now) {
      const elapsed = now - start
      const rise = Math.min(1, elapsed / 1100)
      const settle = elapsed > 2200 ? Math.max(0.35, 1 - (elapsed - 2200) / 1600) : 1
      const env = ((1 - Math.cos(rise * Math.PI)) / 2) * settle

      ctx.clearRect(0, 0, w, h)
      const cy = h / 2

      if (now > nextRipple && elapsed < 2600) {
        ripples.push({ r: 0, a: 0.5 })
        nextRipple = now + 700
      }
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i]
        rp.r += 2.4
        rp.a *= 0.985
        if (rp.a < 0.01) { ripples.splice(i, 1); continue }
        ctx.beginPath()
        ctx.arc(w / 2, cy, rp.r, 0, Math.PI * 2)
        ctx.strokeStyle = `rgba(91,233,255,${rp.a})`
        ctx.lineWidth = 1
        ctx.stroke()
      }

      const grad = ctx.createLinearGradient(0, 0, w, 0)
      grad.addColorStop(0, 'rgba(91,233,255,0)')
      grad.addColorStop(0.2, 'rgba(91,233,255,0.9)')
      grad.addColorStop(0.5, 'rgba(138,123,255,1)')
      grad.addColorStop(0.8, 'rgba(180,120,255,0.9)')
      grad.addColorStop(1, 'rgba(180,120,255,0)')

      const baseAmp = Math.min(h, w) * 0.18 * env
      ctx.beginPath()
      for (let x = 0; x <= w; x += 4) {
        const k = x / w
        const win = Math.sin(k * Math.PI)
        const y = cy
          + Math.sin(x * 0.012 + t * 0.05) * baseAmp * win
          + Math.sin(x * 0.027 - t * 0.03) * baseAmp * 0.45 * win
          + Math.sin(x * 0.06 + t * 0.08) * baseAmp * 0.2 * win
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.strokeStyle = grad
      ctx.lineWidth = 2.4
      ctx.shadowColor = 'rgba(138,123,255,0.8)'
      ctx.shadowBlur = 22
      ctx.stroke()
      ctx.globalAlpha = 0.25
      ctx.stroke()
      ctx.globalAlpha = 1
      ctx.shadowBlur = 0

      t += 1
      raf = requestAnimationFrame(draw)
    }

    if (reduce) {
      setShowText(true)
      const to = setTimeout(finish, 1400)
      return () => { clearTimeout(to); window.removeEventListener('resize', resize) }
    }

    raf = requestAnimationFrame(draw)
    const textTimer = setTimeout(() => setShowText(true), 1200)
    const endTimer = setTimeout(finish, 3800)

    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(textTimer)
      clearTimeout(endTimer)
      window.removeEventListener('resize', resize)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className={'intro' + (done ? ' done' : '')} onClick={finish}>
      <canvas ref={ref} className="intro-canvas" aria-hidden="true" />
      <div className="intro-content">
        <h1 className={'intro-title' + (showText ? ' go' : '')}>IA ANALYZER</h1>
        <div className={'intro-sub' + (showText ? ' go' : '')}>Human · AI-modified · Pure AI</div>
      </div>
      <button className="intro-skip" onClick={finish}>Skip intro</button>
    </div>
  )
}
