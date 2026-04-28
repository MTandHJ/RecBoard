import { useMemo, useState } from "react";
import { ConfigProvider, Empty, Flex, Typography } from "antd";
import DatasetSelector from "./components/DatasetSelector";
import DatasetInfo from "./components/DatasetInfo";
import TagFilter from "./components/TagFilter";
import LeaderboardTable from "./components/LeaderboardTable";
import { results, datasetMetas } from "./data/results";
import colors from "./theme";

const { Title } = Typography;

function App() {
  const datasets = useMemo(
    () => [...new Set(results.map((r) => r.dataset))].sort(),
    [],
  );

  const [selectedDataset, setSelectedDataset] = useState<string>(
    datasets[0] ?? "",
  );
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  const filteredByDataset = useMemo(
    () => results.filter((r) => r.dataset === selectedDataset),
    [selectedDataset],
  );

  const allTags = useMemo(
    () => [...new Set(filteredByDataset.flatMap((r) => r.tags))].sort(),
    [filteredByDataset],
  );

  const filteredData = useMemo(() => {
    if (selectedTags.length === 0) return filteredByDataset;
    return filteredByDataset.filter((r) =>
      selectedTags.some((tag) => r.tags.includes(tag)),
    );
  }, [filteredByDataset, selectedTags]);

  const meta = datasetMetas[selectedDataset];

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: colors.accent,
          colorBorder: colors.border,
          colorBgContainer: colors.bgCard,
          colorText: colors.textPrimary,
        },
        components: {
          Table: {
            cellPaddingBlock: 10,
            cellPaddingInline: 12,
            borderColor: colors.border,
            headerBg: colors.bgHeader,
            headerColor: colors.textInverse,
            headerSortActiveBg: colors.bgHeader,
            headerSortHoverBg: "#5a7284",
            headerFilterHoverBg: "#5a7284",
          },
        },
      }}
    >
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "32px 24px", backgroundColor: colors.bgPage, minHeight: "100vh" }}>
        <Title
          level={1}
          style={{
            textAlign: "center",
            marginBottom: 28,
            fontSize: 36,
            color: colors.textPrimary,
          }}
        >
          🏆 RecBoard
        </Title>
        {datasets.length === 0 ? (
          <Empty description="No benchmark data found" />
        ) : (
          <>
            <Flex
              gap={12}
              style={{ marginBottom: 16 }}
              wrap="wrap"
              align="center"
            >
              <DatasetSelector
                datasets={datasets}
                value={selectedDataset}
                onChange={(v) => {
                  setSelectedDataset(v);
                  setSelectedTags([]);
                }}
              />
              <TagFilter
                tags={allTags}
                value={selectedTags}
                onChange={setSelectedTags}
              />
            </Flex>
            {meta && <DatasetInfo meta={meta} />}
            <LeaderboardTable data={filteredData} meta={meta} />
          </>
        )}
      </div>
    </ConfigProvider>
  );
}

export default App;
