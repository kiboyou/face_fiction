
import { NextRequest, NextResponse } from "next/server";

export const config = {
  api: {
    bodyParser: false,
    sizeLimit: "100mb",
  },
};

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000/api/predict";

export async function POST(req: NextRequest) {
  try {
    // On forward tous les headers sauf host et content-length
    const headers = new Headers();
    req.headers.forEach((value, key) => {
      if (key !== "host" && key !== "content-length") {
        headers.set(key, value);
      }
    });


    const fetchRes = await fetch(BACKEND_URL, {
      method: "POST",
      headers,
      body: req.body,
      duplex: "half", // Important pour le streaming dans Node.js 18+ (App Router)
    } as any);

    // Forward la réponse du backend (status, headers, body)
    const resHeaders = new Headers();
    fetchRes.headers.forEach((v, k) => resHeaders.set(k, v));
    return new NextResponse(fetchRes.body, {
      status: fetchRes.status,
      headers: resHeaders,
    });
  } catch (err: any) {
    // Log l'erreur côté serveur
    console.error("Proxy /api/predict error:", err);
    return NextResponse.json({
      error: "Proxy error: " + (err?.message || err?.toString() || "Unknown error"),
    }, { status: 502 });
  }
}
