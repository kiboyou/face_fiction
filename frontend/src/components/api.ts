
// Utility to fetch metrics from /api/reports
export async function fetchMetrics() {
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
