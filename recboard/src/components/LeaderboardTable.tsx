import { useMemo, useState } from "react";
import { Button, Flex, Table, Tag } from "antd";
import type { ColumnsType } from "antd/es/table";
import type { AggregatedResult, DatasetMeta, MetricStat } from "../types";
import ExpandedRow from "./ExpandedRow";
import colors from "../theme";

interface LeaderboardTableProps {
  data: AggregatedResult[];
  meta?: DatasetMeta;
}

function isBestValue(stat: MetricStat | undefined, bestVal: number): boolean {
  return !!stat && Math.abs(stat.mean - bestVal) < 1e-10;
}

function renderMetric(stat: MetricStat | undefined, isBest: boolean) {
  if (!stat) return "-";
  const mean = stat.mean.toFixed(4);
  const Wrapper = isBest ? "strong" : "span";
  if (stat.std === 0 || isNaN(stat.std)) {
    return <Wrapper>{mean}</Wrapper>;
  }
  return (
    <Wrapper>
      {mean}
      <sub style={{ fontSize: "0.75em", opacity: 0.55 }}>{"\u00b1"}{stat.std.toFixed(4)}</sub>
    </Wrapper>
  );
}

function getMetricKeys(data: AggregatedResult[]): string[] {
  const keySet = new Set<string>();
  for (const item of data) {
    for (const key of Object.keys(item.bestMetrics)) {
      keySet.add(key);
    }
  }
  return Array.from(keySet);
}

function getBestPerMetric(
  data: AggregatedResult[],
  metricKeys: string[],
): Record<string, number> {
  const best: Record<string, number> = {};
  for (const key of metricKeys) best[key] = -Infinity;
  for (const item of data) {
    for (const key of metricKeys) {
      const val = item.bestMetrics[key]?.mean ?? -Infinity;
      if (val > best[key]) best[key] = val;
    }
  }
  return best;
}

const tagStyle: React.CSSProperties = {
  backgroundColor: "transparent",
  color: colors.tag,
  border: `1px solid ${colors.tag}`,
  margin: 0,
};

const bestCellProps = { style: { backgroundColor: colors.bgHighlight } };
const emptyCellProps = {};

const getRowKey = (record: AggregatedResult) => record.id;

const expandable = {
  expandedRowRender: (record: AggregatedResult) => <ExpandedRow record={record} />,
};

function LeaderboardTable({ data, meta }: LeaderboardTableProps) {
  const [showAllMetrics, setShowAllMetrics] = useState(false);
  const metricKeys = useMemo(() => getMetricKeys(data), [data]);
  const bestValues = useMemo(
    () => getBestPerMetric(data, metricKeys),
    [data, metricKeys],
  );

  const coreMetrics = useMemo(() => meta?.core_metrics ?? [], [meta?.core_metrics]);
  const sortBy = meta?.sort_by;
  const hasCoreFilter = coreMetrics.length > 0;
  const extraMetrics = hasCoreFilter
    ? metricKeys.filter((k) => !coreMetrics.includes(k))
    : [];

  const visibleMetricKeys = useMemo(() => {
    if (!hasCoreFilter || showAllMetrics) return metricKeys;
    return metricKeys.filter((k) => coreMetrics.includes(k));
  }, [metricKeys, hasCoreFilter, showAllMetrics, coreMetrics]);

  const columns = useMemo<ColumnsType<AggregatedResult>>(() => [
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      fixed: "left",
      width: 140,
      align: "center",
      sorter: (a, b) => a.model.localeCompare(b.model),
      render: (model: string) => (
        <a href={`https://github.com/MTandHJ/RecBoard/tree/master/${model}`} target="_blank" rel="noopener noreferrer">
          {model}
        </a>
      ),
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
      width: 200,
      align: "center",
    },
    {
      title: "Tags",
      dataIndex: "tags",
      key: "tags",
      width: 150,
      align: "center",
      render: (tags: string[]) => (
        <Flex gap={6} wrap="wrap" justify="center">
          {tags.map((tag) => (
            <Tag key={tag} style={tagStyle}>
              {tag}
            </Tag>
          ))}
        </Flex>
      ),
    },
    ...visibleMetricKeys.map((key) => ({
      title: key,
      key,
      width: 180,
      align: "center" as const,
      sorter: (a: AggregatedResult, b: AggregatedResult) =>
        (a.bestMetrics[key]?.mean ?? 0) - (b.bestMetrics[key]?.mean ?? 0),
      defaultSortOrder:
        key === sortBy ? ("descend" as const) : undefined,
      onCell: (record: AggregatedResult) =>
        isBestValue(record.bestMetrics[key], bestValues[key]) ? bestCellProps : emptyCellProps,
      render: (_: unknown, record: AggregatedResult) =>
        renderMetric(record.bestMetrics[key], isBestValue(record.bestMetrics[key], bestValues[key])),
    })),
  ], [visibleMetricKeys, bestValues, sortBy]);

  return (
    <>
      {hasCoreFilter && extraMetrics.length > 0 && (
        <div style={{ marginBottom: 12, textAlign: "right" }}>
          <Button
            size="small"
            type={showAllMetrics ? "primary" : "default"}
            onClick={() => setShowAllMetrics(!showAllMetrics)}
          >
            {showAllMetrics ? "Show Core Metrics" : `Show All Metrics (+${extraMetrics.length})`}
          </Button>
        </div>
      )}
      <div style={{ border: `2px solid ${colors.border}`, borderRadius: 8, overflow: "hidden" }}>
        <Table
          columns={columns}
          dataSource={data}
          rowKey={getRowKey}
          expandable={expandable}
          scroll={{ x: "max-content" }}
          pagination={false}
          bordered
          size="middle"
        />
      </div>
    </>
  );
}

export default LeaderboardTable;
