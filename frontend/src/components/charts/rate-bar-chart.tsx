'use client';

import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatNumber, formatPercent } from '@/lib/format';
import { axisProps, CHART, gridProps, SERIES } from './chart-theme';
import { chartTooltip } from './chart-tooltip';

export interface RateDatum {
  label: string;
  fraud_rate: number;
  transactions: number;
  frauds: number;
}

interface RateBarChartProps {
  data: RateDatum[];
  /** `bars` = horizontal bars for named categories, `columns` = vertical for ordered values. */
  orientation?: 'bars' | 'columns';
  /** Width reserved for category names when orientation is `bars`. */
  labelWidth?: number;
  /** Print the value at the bar tip (only sensible for a handful of bars). */
  showValues?: boolean;
}

const tooltip = chartTooltip({
  formatValue: (v) => formatPercent(v, 2),
  extra: (row) => [
    ['Transactions', formatNumber(row.transactions as number)],
    ['Frauds', formatNumber(row.frauds as number)],
  ],
});

const tickPercent = (v: number) => formatPercent(v, v < 0.1 ? 1 : 0);

/** Fraud rate per category. One series, so the card title names it and there is no legend. */
export function RateBarChart({ data, orientation = 'bars', labelWidth = 96, showValues = false }: RateBarChartProps) {
  const bars = orientation === 'bars';
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout={bars ? 'vertical' : 'horizontal'} margin={{ top: 8, right: bars ? 48 : 8, bottom: 0, left: 0 }} barCategoryGap="28%">
        <CartesianGrid {...gridProps} horizontal={!bars} vertical={bars} />
        {bars ? (
          <>
            <XAxis type="number" tickFormatter={tickPercent} {...axisProps} axisLine={false} />
            <YAxis type="category" dataKey="label" width={labelWidth} {...axisProps} interval={0} />
          </>
        ) : (
          <>
            <XAxis type="category" dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={12} />
            <YAxis type="number" tickFormatter={tickPercent} width={44} {...axisProps} axisLine={false} />
          </>
        )}
        <Tooltip content={tooltip} cursor={{ fill: CHART.grid, opacity: 0.4 }} />
        <Bar dataKey="fraud_rate" name="Fraud rate" fill={SERIES.primary} radius={bars ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
          {showValues ? (
            <LabelList dataKey="fraud_rate" position={bars ? 'right' : 'top'} formatter={(v) => formatPercent(Number(v), 1)} fill={CHART.tick} fontSize={11} />
          ) : null}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
