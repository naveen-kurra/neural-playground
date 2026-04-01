import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "validator",
          include: ["packages/validator/src/**/*.test.ts"],
          environment: "node"
        }
      },
      {
        test: {
          name: "exporter",
          include: ["packages/exporter-pytorch/src/**/*.test.ts"],
          environment: "node"
        }
      },
      {
        test: {
          name: "web",
          include: ["apps/web/src/**/*.test.ts", "apps/web/src/**/*.test.tsx"],
          environment: "jsdom",
          globals: true,
          setupFiles: ["apps/web/src/test-setup.ts"]
        }
      }
    ]
  }
});
