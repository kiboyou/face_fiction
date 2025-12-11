// Utility to fetch models from the backend
export async function fetchModels() {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error("Erreur lors de la récupération des modèles");
  return res.json();
}
