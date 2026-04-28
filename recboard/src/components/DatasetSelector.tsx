import { Select } from "antd";

interface DatasetSelectorProps {
  datasets: string[];
  value: string;
  onChange: (value: string) => void;
}

function DatasetSelector({ datasets, value, onChange }: DatasetSelectorProps) {
  return (
    <Select
      value={value}
      onChange={onChange}
      style={{ minWidth: 280 }}
      options={datasets.map((d) => ({ label: d, value: d }))}
    />
  );
}

export default DatasetSelector;