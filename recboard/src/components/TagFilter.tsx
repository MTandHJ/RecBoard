import { Select } from "antd";

interface TagFilterProps {
  tags: string[];
  value: string[];
  onChange: (value: string[]) => void;
}

function TagFilter({ tags, value, onChange }: TagFilterProps) {
  if (tags.length === 0) return null;
  return (
    <Select
      mode="multiple"
      allowClear
      placeholder="Filter by tags"
      value={value}
      onChange={onChange}
      style={{ minWidth: 200 }}
      options={tags.map((t) => ({ label: t, value: t }))}
    />
  );
}

export default TagFilter;