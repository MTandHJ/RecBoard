export interface Run {
  id: string;
  params: { config: string; seed: number };
  metrics: {
    train: Record<string, number>;
    valid: Record<string, number>;
    test: Record<string, number>;
    best: Record<string, number>;
  };
}

export interface MetricStat {
  mean: number;
  std: number;
}

export interface DatasetMeta {
  description: string;
  build_command: string;
  sort_by: string;
  core_metrics: string[];
  statistics: Record<string, number>;
}

export interface AggregatedResult {
  model: string;
  description: string;
  dataset: string;
  tags: string[];
  bestMetrics: Record<string, MetricStat>;
  runs: Run[];
  config: Record<string, unknown>;
  timestamp: string;
}