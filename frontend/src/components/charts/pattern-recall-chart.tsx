'use client';

import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatNumber, formatPercent } from '@/lib/format';
import { axisProps, CHART, gridProps, legendProps, SERIES } from './chart-theme';
import { chartTooltip } from './chart-tooltip';

export interface PatternDatum {
  label: string;
  /** null when the pattern did not exist in the baseline window. */
  baseline: number | null;
  current: number;
  frauds: number;
}

const tooltip = chartTooltip({
  formatValue: (v) => formatPercent(v, 1),
  extra: (row) => [['Frauds this month', formatNumber(row.frauds as number)]],
});

/** Recall per fraud pattern for the selected month against the June hold-out. */
export function PatternRecallChart({ data, currentLabel }: { data: PatternDatum[]; currentLabel: string }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 0, left: 0 }} barCategoryGap="24%" barGap={2}>
        <CartesianGrid {...gridProps} horizontal={false} />
        <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => formatPercent(v, 0)} {...axisProps} axisLine={false} />
        <YAxis type="category" dataKey="label" width={150} {...axisProps} interval={0} />
        <Tooltip content={tooltip} cursor={{ fill: CHART.grid, opacity: 0.4 }} />
        <Legend verticalAlign="top" align="right" height={28} {...legendProps} />
        <Bar dataKey="baseline" name="June baseline" fill={SERIES.neutral} radius={[0, 3, 3, 0]} maxBarSize={12} isAnimationActive={false} />
        <Bar dataKey="current" name={currentLabel} fill={SERIES.primary} radius={[0, 3, 3, 0]} maxBarSize={12} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
