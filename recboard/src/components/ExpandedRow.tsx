import { Descriptions, Table } from "antd";
import type { AggregatedResult } from "../types";

interface ExpandedRowProps {
  record: AggregatedResult;
}

function ExpandedRow({ record }: ExpandedRowProps) {
  const { runs, config } = record;

  const metricKeys = runs.length > 0 ? Object.keys(runs[0].metrics.best) : [];

  const seedColumns = [
    {
      title: "Seed",
      dataIndex: "seed",
      key: "seed",
      width: 80,
    },
    ...metricKeys.map((key) => ({
      title: key,
      dataIndex: key,
      key,
      render: (v: number) => v?.toFixed(4) ?? "-",
    })),
  ];

  const seedData = runs.map((run) => ({
    key: run.id,
    seed: run.params.seed,
    ...run.metrics.best,
  }));

  const configEntries = Object.entries(config);

  return (
    <div style={{ padding: "0 16px" }}>
      <h4 style={{ marginTop: 0 }}>Results per Seed</h4>
      <Table
        columns={seedColumns}
        dataSource={seedData}
        pagination={false}
        size="small"
        bordered
        scroll={{ x: "max-content" }}
      />
      {configEntries.length > 0 && (
        <>
          <h4>Config</h4>
          <Descriptions
            size="small"
            bordered
            column={3}
            styles={{ label: { backgroundColor: "#CBAF86", fontWeight: 500, color: "#fff" } }}
          >
            {configEntries.map(([key, value]) => (
              <Descriptions.Item key={key} label={key}>
                {typeof value === "object" ? JSON.stringify(value) : String(value)}
              </Descriptions.Item>
            ))}
          </Descriptions>
        </>
      )}
    </div>
  );
}

export default ExpandedRow;