import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './styles.css'

// Enable entrance animations only when the timeline is actually running.
// If requestAnimationFrame never fires (e.g. background/throttled tab), content
// stays visible instead of being stuck at opacity:0.
if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  requestAnimationFrame(() => document.documentElement.classList.add('anim-on'))
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
