import React, { useEffect, useRef } from 'react'

/**
 * Adds the `in` class to every `.reveal` inside the returned ref once it
 * scrolls into view, with a small stagger. Attach the ref to a wrapper:
 *   const reveal = useReveal()
 *   return <div ref={reveal}>...<div className="reveal">...</div></div>
 */
export default function useReveal() {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in')
            io.unobserve(e.target)
          }
        })
      },
      { threshold: 0.15, rootMargin: '0px 0px -8% 0px' }
    )

    const items = el.querySelectorAll('.reveal')
    items.forEach((it, i) => {
      if (!it.style.transitionDelay) it.style.transitionDelay = `${Math.min(i * 70, 420)}ms`
      io.observe(it)
    })

    return () => io.disconnect()
  }, [])

  return ref
}
