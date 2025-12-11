import { NextRequest } from "next/server";

// Proxy /api/reports/[report_name] to FastAPI backend
export async function GET(req: NextRequest, { params }: { params: { report_name: string } }) {
  const backendUrl = process.env.BACKEND_URL || `http://localhost:8000/api/reports/${params.report_name}`;
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
