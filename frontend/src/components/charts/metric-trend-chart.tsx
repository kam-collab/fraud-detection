'use client';

import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatStat } from '@/lib/format';
import type { Interval } from '@/types';
import { axisProps, CHART, gridProps, SERIES } from './chart-theme';
import { chartTooltip } from './chart-tooltip';

export interface TrendPoint {
  /** Short x label, e.g. "Jun" or "Jul". */
  label: string;
  value: number;
}

interface MetricTrendChartProps {
  name: string;
  data: TrendPoint[];
  baseline: number;
  /** 95% confidence interval of the baseline, drawn as a band. */
  ci?: Interval;
  /** Label of the point to emphasise (the selected month). */
  highlight?: string;
}

/** One metric over time against its release baseline and the baseline's 95% CI band. */
export function MetricTrendChart({ name, data, baseline, ci, highlight }: MetricTrendChartProps) {
  const tooltip = chartTooltip({ formatValue: (v) => formatStat(v, 3), extra: () => [['Baseline', formatStat(baseline, 3)]] });
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...gridProps} vertical={false} />
        <XAxis dataKey="label" {...axisProps} padding={{ left: 16, right: 16 }} />
        <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} width={36} {...axisProps} axisLine={false} />
        {ci ? <ReferenceArea y1={ci[0]} y2={ci[1]} fill={SERIES.neutral} fillOpacity={0.18} stroke="none" /> : null}
        <ReferenceLine y={baseline} stroke={SERIES.neutral} strokeWidth={1.5} />
        <Tooltip content={tooltip} cursor={{ stroke: CHART.axis }} />
        <Line
          type="linear"
          dataKey="value"
          name={name}
          stroke={SERIES.primary}
          strokeWidth={2}
          isAnimationActive={false}
          dot={({ cx, cy, payload, index }) => {
            const active = (payload as TrendPoint).label === highlight;
            return <circle key={index} cx={cx} cy={cy} r={active ? 6 : 4} fill={SERIES.primary} stroke={active ? CHART.ink : CHART.surface} strokeWidth={2} />;
          }}
          activeDot={{ r: 5, stroke: CHART.surface, strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
