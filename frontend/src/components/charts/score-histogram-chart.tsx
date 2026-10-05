'use client';

import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatPercent } from '@/lib/format';
import { axisProps, CHART, gridProps, legendProps, SERIES } from './chart-theme';
import { chartTooltip } from './chart-tooltip';

export interface HistogramBin {
  /** "0.10–0.15" */
  bin: string;
  baseline: number;
  current: number;
}

interface ScoreHistogramChartProps {
  data: HistogramBin[];
  currentLabel: string;
}

const FLOOR = 1e-4;
const ticks = [0.0001, 0.001, 0.01, 0.1, 1];
const formatShare = (v: number) => (v < 0.001 ? formatPercent(v, 2) : v < 0.01 ? formatPercent(v, 1) : formatPercent(v, 0));

const tooltip = chartTooltip({
  formatValue: (v) => formatPercent(v, v < 0.01 ? 3 : 2),
  formatLabel: (label) => `Score ${label}`,
});

/**
 * Share of transactions per score bin, month vs the June baseline.
 * The y axis is logarithmic: about 97% of payments score below 0.05, so on a
 * linear axis every other bin would be invisible.
 */
export function ScoreHistogramChart({ data, currentLabel }: ScoreHistogramChartProps) {
  // A log axis cannot draw zero; clamp empty bins to the axis floor.
  const safe = data.map((d) => ({ ...d, baseline: Math.max(d.baseline, FLOOR), current: Math.max(d.current, FLOOR) }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={safe} margin={{ top: 8, right: 8, bottom: 16, left: 0 }} barCategoryGap="20%" barGap={2}>
        <CartesianGrid {...gridProps} vertical={false} />
        <XAxis
          dataKey="bin"
          {...axisProps}
          interval={1}
          tickFormatter={(bin: string) => bin.split('–')[0]}
          label={{ value: 'Fraud score (bin start)', position: 'insideBottom', offset: -10, fill: CHART.tick, fontSize: 11 }}
        />
        <YAxis scale="log" domain={[FLOOR, 1]} ticks={ticks} allowDataOverflow tickFormatter={formatShare} width={52} {...axisProps} axisLine={false} />
        <Tooltip content={tooltip} cursor={{ fill: CHART.grid, opacity: 0.4 }} />
        <Legend verticalAlign="top" align="right" height={28} {...legendProps} />
        <Bar dataKey="baseline" name="June baseline" fill={SERIES.neutral} radius={[3, 3, 0, 0]} maxBarSize={14} isAnimationActive={false} />
        <Bar dataKey="current" name={currentLabel} fill={SERIES.primary} radius={[3, 3, 0, 0]} maxBarSize={14} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
