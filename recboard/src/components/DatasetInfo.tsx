import { Card, Flex, Statistic } from "antd";
import type { DatasetMeta } from "../types";
import colors from "../theme";

interface DatasetInfoProps {
  meta: DatasetMeta;
}

function DatasetInfo({ meta }: DatasetInfoProps) {
  const statEntries = Object.entries(meta.statistics);

  return (
    <Card
      size="small"
      style={{ marginBottom: 16, borderColor: colors.border }}
      styles={{ body: { padding: "16px 20px" } }}
    >
      {meta.description && (
        <div style={{ marginBottom: 14, color: colors.textSecondary, fontSize: 14 }}>
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
                backgroundColor: colors.bgStat,
                borderColor: colors.borderStat,
                minWidth: 130,
              }}
              styles={{ body: { padding: "10px 16px" } }}
            >
              <Statistic
                title={<span style={{ fontSize: 12, color: colors.textTertiary }}>{key}</span>}
                value={value}
                valueStyle={{ fontSize: 20, fontWeight: 600, color: colors.textPrimary }}
              />
            </Card>
          ))}
        </Flex>
      )}

      {meta.build_command && (
        <div
          style={{
            backgroundColor: colors.bgCode,
            borderRadius: 6,
            padding: "10px 16px",
            fontFamily: "'Menlo', 'Consolas', 'Courier New', monospace",
            fontSize: 13,
            color: colors.textCode,
            overflowX: "auto",
          }}
        >
          <span style={{ color: colors.accentGreen }}>$</span>{" "}
          {meta.build_command}
        </div>
      )}
    </Card>
  );
}

export default DatasetInfo;
