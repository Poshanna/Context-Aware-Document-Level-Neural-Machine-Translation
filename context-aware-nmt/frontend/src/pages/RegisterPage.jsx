import { useState } from 'react'
import { Link } from 'react-router-dom'

export default function RegisterPage() {
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)

  return (
    <div className="max-w-md mx-auto mt-12">
      <div className="card">
        <div className="card-header">
          <h1 className="text-xl font-bold text-slate-900">Create an account</h1>
          <p className="text-sm text-slate-500 mt-1">
            Passwords are hashed with argon2. No raw secrets are stored.
          </p>
        </div>
        <div className="card-body space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Full name (optional)</label>
            <input
              type="text"
              value={fullName}
              onChange={e => setFullName(e.target.value)}
              className="input"
              placeholder="Jane Doe"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              className="input"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Password (min 8 chars)</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="input"
              placeholder="••••••••"
              minLength={8}
            />
          </div>
          <button className="btn-primary w-full" disabled={loading}>
            Create account
          </button>
          <div className="text-sm text-center text-slate-500">
            Already registered?{' '}
            <Link to="/login" className="text-brand-700 hover:underline">
              Sign in
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
