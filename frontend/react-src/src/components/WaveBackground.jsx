import React, { useEffect, useRef } from 'react'

function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const rect = canvas.getBoundingClientRect()
  canvas.width = Math.max(1, Math.floor(rect.width * dpr))
  canvas.height = Math.max(1, Math.floor(rect.height * dpr))
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { ctx, w: rect.width, h: rect.height }
}

/** Ambient flowing waves rendered behind the whole app. */
export default function WaveBackground() {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    let raf, w, h, ctx
    let t = 0
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    function resize() {
      const s = setupCanvas(canvas)
      ctx = s.ctx; w = s.w; h = s.h
    }
    resize()
    window.addEventListener('resize', resize)

    // slow, long "S"-shaped curves (one gentle bend across the screen)
    const layers = [
      { amp: 80, len: 1.05, speed: 0.0035, y: 0.5, col: 'rgba(91,233,255,0.26)', lw: 1.6 },
      { amp: 100, len: 0.85, speed: -0.0025, y: 0.56, col: 'rgba(138,123,255,0.22)', lw: 1.6 },
    ]

    function draw() {
      ctx.clearRect(0, 0, w, h)
      layers.forEach((L, idx) => {
        ctx.beginPath()
        for (let x = 0; x <= w; x += 6) {
          const phase = (x / w) * Math.PI * 2 * L.len
          const y = h * L.y + Math.sin(phase + t * L.speed + idx) * L.amp
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        }
        ctx.strokeStyle = L.col
        ctx.lineWidth = L.lw
        ctx.shadowColor = L.col
        ctx.shadowBlur = 16
        ctx.stroke()
      })
      ctx.shadowBlur = 0
    }

    function loop() {
      t += 1
      draw()
      raf = requestAnimationFrame(loop)
    }
    if (reduce) draw(); else loop()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={ref} className="wave-canvas" aria-hidden="true" />
}
