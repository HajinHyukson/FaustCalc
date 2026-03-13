import path from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow tracing files outside frontend/ when deployed from a monorepo root.
  outputFileTracingRoot: path.join(process.cwd(), ".."),
  outputFileTracingIncludes: {
    "/api/portfolio": [
      "../src/**/*",
      "../requirements.txt",
    ],
  },
};

export default nextConfig;
