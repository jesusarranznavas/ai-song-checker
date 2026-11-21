import React, { useState } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import WaveBackground from './components/WaveBackground.jsx'
import Navbar from './components/Navbar.jsx'
import Footer from './components/Footer.jsx'
import Intro from './components/Intro.jsx'
import Home from './pages/Home.jsx'
import Analyzer from './pages/Analyzer.jsx'
import About from './pages/About.jsx'

export default function App() {
  const location = useLocation()
  const [intro, setIntro] = useState(() => !sessionStorage.getItem('introSeen'))

  function endIntro() {
    sessionStorage.setItem('introSeen', '1')
    setIntro(false)
  }

  // Scroll to top on route change
  React.useEffect(() => { window.scrollTo(0, 0) }, [location.pathname])

  return (
    <>
      <div className="app-bg" aria-hidden="true" />
      <WaveBackground />
      <div className="grain" aria-hidden="true" />

      <div className="page-wrap">
        <Navbar />
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyzer" element={<Analyzer />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
        <Footer />
      </div>

      {intro && <Intro onDone={endIntro} />}
    </>
  )
}
