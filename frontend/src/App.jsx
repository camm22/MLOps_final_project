import { useState } from 'react'
import './App.css'

const SPECIES_MAP = {
  0: { name: 'Setosa', emoji: '🌸' },
  1: { name: 'Versicolor', emoji: '🌺' },
  2: { name: 'Virginica', emoji: '🌹' },
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [features, setFeatures] = useState({
    sepalLength: '',
    sepalWidth: '',
    petalLength: '',
    petalWidth: '',
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: e.target.value })
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const values = [
      parseFloat(features.sepalLength),
      parseFloat(features.sepalWidth),
      parseFloat(features.petalLength),
      parseFloat(features.petalWidth),
    ]

    if (values.some(isNaN)) {
      setError('Please fill in all fields with valid numbers.')
      setLoading(false)
      return
    }

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: values }),
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data.prediction)
    } catch (err) {
      setError(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const fillExample = () => {
    setFeatures({
      sepalLength: '5.1',
      sepalWidth: '3.5',
      petalLength: '1.4',
      petalWidth: '0.2',
    })
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🌿 Iris Classifier</h1>
        <p className="subtitle">MLOps Project — Iris Species Prediction</p>
      </header>

      <main className="main">
        <form className="form" onSubmit={handlePredict}>
          <div className="form-grid">
            <div className="input-group">
              <label htmlFor="sepalLength">Sepal Length (cm)</label>
              <input
                id="sepalLength"
                name="sepalLength"
                type="number"
                step="0.1"
                placeholder="ex: 5.1"
                value={features.sepalLength}
                onChange={handleChange}
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="sepalWidth">Sepal Width (cm)</label>
              <input
                id="sepalWidth"
                name="sepalWidth"
                type="number"
                step="0.1"
                placeholder="ex: 3.5"
                value={features.sepalWidth}
                onChange={handleChange}
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="petalLength">Petal Length (cm)</label>
              <input
                id="petalLength"
                name="petalLength"
                type="number"
                step="0.1"
                placeholder="ex: 1.4"
                value={features.petalLength}
                onChange={handleChange}
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="petalWidth">Petal Width (cm)</label>
              <input
                id="petalWidth"
                name="petalWidth"
                type="number"
                step="0.1"
                placeholder="ex: 0.2"
                value={features.petalWidth}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          <div className="buttons">
            <button type="submit" className="btn-predict" disabled={loading}>
              {loading ? '⏳ Predicting...' : '🔍 Predict'}
            </button>
            <button type="button" className="btn-example" onClick={fillExample}>
              📝 Example
            </button>
          </div>
        </form>

        {error && (
          <div className="result error">
            <p>❌ {error}</p>
          </div>
        )}

        {result !== null && (
          <div className="result success">
            <div className="prediction-emoji">{SPECIES_MAP[result]?.emoji}</div>
            <h2>{SPECIES_MAP[result]?.name || `Class ${result}`}</h2>
            <p className="prediction-detail">
              Predicted class: <strong>{result}</strong>
            </p>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>MLOps Final Project — FastAPI + React + MLflow + DVC + GitHub Actions</p>
      </footer>
    </div>
  )
}

export default App
