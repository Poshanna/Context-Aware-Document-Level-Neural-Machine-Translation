import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import HomePage from './pages/HomePage.jsx'
import TranslatePage from './pages/TranslatePage.jsx'
import DocumentPage from './pages/DocumentPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import EvaluatePage from './pages/EvaluatePage.jsx'
import ComparePage from './pages/ComparePage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import RegisterPage from './pages/RegisterPage.jsx'

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/translate" element={<TranslatePage />} />
          <Route path="/document" element={<DocumentPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/evaluate" element={<EvaluatePage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="border-t border-slate-200 bg-white py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          Context-Aware Document-Level NMT for Indian Languages — B.E. AI &amp; ML Research Project
        </div>
      </footer>
    </div>
  )
}

export default App
