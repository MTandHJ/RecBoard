import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "fs";
import { join, basename, resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..", "..");
const BENCHMARK_DIR = join(ROOT, "benchmark");
const OUTPUT_FILE = join(__dirname, "..", "src", "data", "results.ts");

const CONFIG_BLACKLIST = new Set([
  "CHECKPOINT_PATH",
  "LOG_PATH",
  "SAVED_FILENAME",
  "CHECKPOINT_FILENAME",
  "BEST_FILENAME",
  "MONITOR_FILENAME",
  "MONITOR_BEST_FILENAME",
  "CONFIG_FILENAME",
  "SUMMARY_DIR",
  "SUMMARY_FILENAME",
  "RESULTS_FILENAME",
  "DATA_DIR",
  "device",
  "ddp_backend",
  "num_workers",
  "id",
  "resume",
  "benchmark",
  "eval_test",
  "eval_valid",
  "log2console",
  "log2file",
  "retain_seen",
  "early_stop_patience",
  "CHECKPOINT_FREQ",
  "CHECKPOINT_MODULES",
]);

function filterConfig(config) {
  const filtered = {};
  for (const [key, value] of Object.entries(config)) {
    if (!CONFIG_BLACKLIST.has(key)) {
      filtered[key] = value;
    }
  }
  return filtered;
}

function computeStats(values) {
  const n = values.length;
  if (n === 0) return { mean: 0, std: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  return { mean, std: Math.sqrt(variance) };
}

function aggregateRuns(runs) {
  if (runs.length === 0) return {};
  const metricKeys = Object.keys(runs[0].metrics.best || {});
  const result = {};
  for (const key of metricKeys) {
    const values = runs.map((r) => r.metrics.best[key]).filter((v) => v != null);
    result[key] = computeStats(values);
  }
  return result;
}

function main() {
  if (!existsSync(BENCHMARK_DIR)) {
    console.log("No benchmark/ directory found, generating empty data.");
    writeFileSync(
      OUTPUT_FILE,
      'import type { AggregatedResult } from "../types";\n\nexport const results: AggregatedResult[] = [];\n'
    );
    return;
  }

  const results = [];
  const datasetMetas = {};
  const datasets = readdirSync(BENCHMARK_DIR).filter((d) =>
    statSync(join(BENCHMARK_DIR, d)).isDirectory()
  );

  for (const dataset of datasets) {
    const datasetDir = join(BENCHMARK_DIR, dataset);

    const metaPath = join(datasetDir, "meta.json");
    if (existsSync(metaPath)) {
      datasetMetas[dataset] = JSON.parse(readFileSync(metaPath, "utf-8"));
    }

    const files = readdirSync(datasetDir).filter(
      (f) => f.endsWith(".json") && f !== "meta.json"
    );

    for (const file of files) {
      const model = basename(file, ".json");
      const content = JSON.parse(readFileSync(join(datasetDir, file), "utf-8"));
      const evaluations = Array.isArray(content) ? content : [content];

      for (const evaluation of evaluations) {
        results.push({
          model,
          description: evaluation.description || "",
          dataset: evaluation.dataset || dataset,
          tags: evaluation.tags || [],
          bestMetrics: aggregateRuns(evaluation.runs || []),
          runs: evaluation.runs || [],
          config: filterConfig(evaluation.config || {}),
          timestamp: evaluation.timestamp || "",
        });
      }
    }
  }

  const output = `import type { AggregatedResult, DatasetMeta } from "../types";

export const results: AggregatedResult[] = ${JSON.stringify(results, null, 2)};

export const datasetMetas: Record<string, DatasetMeta> = ${JSON.stringify(datasetMetas, null, 2)};
`;

  writeFileSync(OUTPUT_FILE, output);
  console.log(`Generated ${results.length} results from ${datasets.length} datasets.`);
}

main();