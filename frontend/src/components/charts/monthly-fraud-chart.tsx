'use client';

import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatMonth, formatNumber, formatPercent } from '@/lib/format';
import { axisProps, CHART, gridProps, SERIES } from './chart-theme';
import { chartTooltip } from './chart-tooltip';

interface MonthlyFraudChartProps {
  data: { month: string; fraud_rate: number; transactions: number; frauds: number }[];
  /** First month after the development period, marked on the chart. */
  liveFrom?: string;
}

const tooltip = chartTooltip({
  formatValue: (v) => formatPercent(v, 2),
  formatLabel: (label) => formatMonth(String(label)),
  extra: (row) => [
    ['Transactions', formatNumber(row.transactions as number)],
    ['Frauds', formatNumber(row.frauds as number)],
  ],
});

export function MonthlyFraudChart({ data, liveFrom }: MonthlyFraudChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 20, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid {...gridProps} vertical={false} />
        <XAxis dataKey="month" tickFormatter={(m: string) => formatMonth(m).slice(0, 3)} {...axisProps} />
        <YAxis tickFormatter={(v: number) => formatPercent(v, 1)} width={44} domain={[0, 'auto']} {...axisProps} axisLine={false} />
        <Tooltip content={tooltip} cursor={{ stroke: CHART.axis }} />
        {liveFrom && data.some((d) => d.month === liveFrom) ? (
          <ReferenceLine x={liveFrom} stroke={CHART.tick} label={{ value: 'Live period', position: 'insideTopLeft', fill: CHART.tick, fontSize: 11, dy: -18 }} />
        ) : null}
        <Line
          type="monotone"
          dataKey="fraud_rate"
          name="Fraud rate"
          stroke={SERIES.primary}
          strokeWidth={2}
          dot={{ r: 4, fill: SERIES.primary, stroke: CHART.surface, strokeWidth: 2 }}
          activeDot={{ r: 5, stroke: CHART.surface, strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
