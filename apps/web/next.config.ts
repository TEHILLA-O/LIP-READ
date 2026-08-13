import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Without this, Turbopack walks up past the repository looking for a lockfile
  // and settles on the home directory as the project root.
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
