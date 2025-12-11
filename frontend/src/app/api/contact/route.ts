export async function POST(request: Request) {
  try {
    const data = await request.json();
    // In a real app, forward to your backend or send an email here.
    console.log("Contact form submission:", data);
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (e: any) {
    return new Response(e?.message || "Invalid request", { status: 400 });
  }
}
