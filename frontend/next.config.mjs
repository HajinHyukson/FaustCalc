import path from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow tracing files outside frontend/ when deployed from a monorepo root.
  outputFileTracingRoot: path.join(process.cwd(), ".."),
  outputFileTracingIncludes: {
    "/api/portfolio/route": [
      "../src/**/*",
      "../requirements.txt",
    ],
    "/api/portfolio": [
      "../src/**/*",
      "../requirements.txt",
    ],
    "/*": [
      "../src/**/*",
      "../requirements.txt",
    ],
  },
};

export default nextConfig;
