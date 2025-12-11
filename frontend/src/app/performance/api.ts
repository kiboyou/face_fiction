// Utility to fetch metrics from the backend
export async function fetchMetrics(model_name?: string) {
  // Always fetch metrics from /api/reports (ignoring model_name)
  const res = await fetch("/api/reports");
  if (!res.ok) throw new Error("Erreur lors de la récupération des métriques");
  return await res.json();
}

// Utility to fetch health (for number of models, etc)
export async function fetchHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Erreur lors de la récupération du statut");
  return res.json();
}
