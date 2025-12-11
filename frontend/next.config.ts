import type { NextConfig } from "next";


const nextConfig: NextConfig = {
  /* config options here */
  api: {
    bodyParser: {
      sizeLimit: "100mb",
    },
  },
};

export default nextConfig;
