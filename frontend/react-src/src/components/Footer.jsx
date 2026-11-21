import React from 'react'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <span>© {new Date().getFullYear()} IA Analyzer — Final Degree Project (TFG)</span>
        <span>Server-side audio intelligence · Human vs AI detection</span>
      </div>
    </footer>
  )
}
