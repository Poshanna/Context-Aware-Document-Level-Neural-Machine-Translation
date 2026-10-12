import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  const navLinkClass = ({ isActive }) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? 'bg-brand-50 text-brand-700'
        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
    }`

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-8">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold">
                N
              </div>
              <span className="font-semibold text-slate-900">ContextNMT</span>
            </Link>
            <div className="hidden md:flex items-center gap-1">
              <NavLink to="/translate" className={navLinkClass}>
                Translate
              </NavLink>
              <NavLink to="/document" className={navLinkClass}>
                Document
              </NavLink>
              <NavLink to="/history" className={navLinkClass}>
                History
              </NavLink>
              <NavLink to="/evaluate" className={navLinkClass}>
                Evaluate
              </NavLink>
              <NavLink to="/compare" className={navLinkClass}>
                Compare
              </NavLink>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link to="/login" className="btn-secondary">
              Sign in
            </Link>
            <Link to="/register" className="btn-primary">
              Register
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}
