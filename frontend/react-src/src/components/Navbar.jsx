import React, { useState } from 'react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import { Menu, X, Waves } from 'lucide-react'

const LINKS = [
  { label: 'Home', path: '/' },
  { label: 'Analyzer', path: '/analyzer' },
  { label: 'About', path: '/about' },
]

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const go = (p) => { navigate(p); setOpen(false) }

  return (
    <>
      <header className="nav">
        <div className="nav-inner">
          <Link to="/" className="brand animate-blur-fade-up" style={{ animationDelay: '0ms' }}>
            <span className="brand-mark"><Waves size={26} style={{ color: 'var(--accent)' }} /></span>
            <span className="brand-text">IA <span className="accent">ANALYZER</span></span>
          </Link>

          <nav className="nav-links">
            {LINKS.map((l, i) => (
              <NavLink
                key={l.path}
                to={l.path}
                end={l.path === '/'}
                className={({ isActive }) => 'nav-link animate-blur-fade-up' + (isActive ? ' active' : '')}
                style={{ animationDelay: `${100 + i * 50}ms` }}
              >
                {l.label}
              </NavLink>
            ))}
          </nav>

          <div className="nav-actions">
            <button className="nav-hamburger liquid-glass icon-btn animate-blur-fade-up" style={{ animationDelay: '300ms' }} aria-label="Menu" onClick={() => setOpen((v) => !v)}>
              <span className="menu-icon">
                <span className={open ? 'hide' : 'show'}><Menu size={20} /></span>
                <span className={open ? 'show' : 'hide'}><X size={20} /></span>
              </span>
            </button>
          </div>
        </div>
      </header>

      <div className={'mobile-menu ' + (open ? 'open' : 'closed')}>
        {LINKS.map((l) => (
          <button key={l.path} className="mobile-link" onClick={() => go(l.path)}>{l.label}</button>
        ))}
      </div>
    </>
  )
}
