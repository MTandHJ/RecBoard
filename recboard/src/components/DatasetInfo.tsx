import { Card, Flex, Statistic } from "antd";
import type { DatasetMeta } from "../types";

interface DatasetInfoProps {
  meta: DatasetMeta;
}

function DatasetInfo({ meta }: DatasetInfoProps) {
  const statEntries = Object.entries(meta.statistics);

  return (
    <Card
      size="small"
      style={{ marginBottom: 16, borderColor: "#e8e8e8" }}
      styles={{ body: { padding: "16px 20px" } }}
    >
      {meta.description && (
        <div style={{ marginBottom: 14, color: "#555", fontSize: 14 }}>
          {meta.description}
        </div>
      )}

      {statEntries.length > 0 && (
        <Flex gap={16} wrap="wrap" style={{ marginBottom: 14 }}>
          {statEntries.map(([key, value]) => (
            <Card
              key={key}
              size="small"
              style={{
                backgroundColor: "#f7f8fa",
                borderColor: "#eee",
                minWidth: 130,
              }}
              styles={{ body: { padding: "10px 16px" } }}
            >
              <Statistic
                title={<span style={{ fontSize: 12, color: "#888" }}>{key}</span>}
                value={value}
                valueStyle={{ fontSize: 20, fontWeight: 600, color: "#333" }}
              />
            </Card>
          ))}
        </Flex>
      )}

      {meta.build_command && (
        <div
          style={{
            backgroundColor: "#1e1e2e",
            borderRadius: 6,
            padding: "10px 16px",
            fontFamily: "'Menlo', 'Consolas', 'Courier New', monospace",
            fontSize: 13,
            color: "#cdd6f4",
            overflowX: "auto",
          }}
        >
          <span style={{ color: "#a6e3a1" }}>$</span>{" "}
          {meta.build_command}
        </div>
      )}
    </Card>
  );
}

export default DatasetInfo;
