// Same contract as your original api.js — talks to the local backend.
export async function health() {
  const res = await fetch(`/api/health`)
  if (!res.ok) throw new Error(await res.text())
  return await res.json()
}

export async function analyzeFile(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`/api/analyze`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || 'API error')
  }

  return await res.json()
}
