import { useState } from 'react'

export default function useJobPolling(jobId, fetchFn, intervalMs = 2000) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)

  // Placeholder — real impl uses useInterval + fetchFn, stops when job.status is terminal
  return { job, setJob, error, setError }
}
