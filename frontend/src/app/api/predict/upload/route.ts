import { NextRequest } from "next/server";

// Proxy /api/predict/upload to FastAPI backend
export async function POST(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000/api/predict/upload";
  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (key !== "host" && key !== "content-length") {
      headers.set(key, value);
    }
  });
  const fetchRes = await fetch(backendUrl, {
    method: "POST",
    headers,
    body: req.body,
    duplex: "half",
  } as any);
  const resHeaders = new Headers();
  fetchRes.headers.forEach((v, k) => resHeaders.set(k, v));
  return new Response(fetchRes.body, {
    status: fetchRes.status,
    headers: resHeaders,
  });
}
