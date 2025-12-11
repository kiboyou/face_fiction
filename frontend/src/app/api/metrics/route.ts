import { NextRequest } from "next/server";

// Proxy /api/metrics to FastAPI backend
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const modelName = searchParams.get("model_name");
  let backendUrl = process.env.BACKEND_URL || "http://localhost:8000/api/metrics";
  if (modelName) {
    backendUrl += `?model_name=${encodeURIComponent(modelName)}`;
  }
  const res = await fetch(backendUrl, {
    method: "GET",
    headers: {
      "accept": "application/json",
    },
    cache: "no-store",
  });
  const data = await res.text();
  return new Response(data, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}
