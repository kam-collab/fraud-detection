'use client';

import { CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatPercent } from '@/lib/format';
import type { PrCurvePoint } from '@/types';
import { axisProps, CHART, gridProps, legendProps, SERIES } from './chart-theme';
import { chartTooltip } from './chart-tooltip';

export interface OperatingPoint {
  label: string;
  recall: number;
  precision: number;
}

interface PrCurveChartProps {
  champion: PrCurvePoint[];
  baseline: PrCurvePoint[];
  /** Champion operating points at the review and block thresholds. */
  operatingPoints: OperatingPoint[];
}

/**
 * Linear interpolation of precision at fixed recall steps, so both models share one
 * x grid and the tooltip can read them side by side. Display only; metrics come from the API.
 */
export function resamplePrCurve(points: PrCurvePoint[], steps = 100): (number | null)[] {
  const sorted = [...points].sort((a, b) => a.recall - b.recall);
  if (sorted.length === 0) return Array.from({ length: steps + 1 }, () => null);
  let j = 0;
  return Array.from({ length: steps + 1 }, (_, i) => {
    const r = i / steps;
    if (r < sorted[0].recall) return null;
    while (j < sorted.length - 1 && sorted[j + 1].recall < r) j += 1;
    const a = sorted[j];
    const b = sorted[Math.min(j + 1, sorted.length - 1)];
    if (b.recall === a.recall || r <= a.recall) return a.precision;
    if (r >= b.recall) return b.precision;
    return a.precision + ((r - a.recall) / (b.recall - a.recall)) * (b.precision - a.precision);
  });
}

const tooltip = chartTooltip({
  formatValue: (v) => formatPercent(v, 1),
  formatLabel: (label) => `Recall ${formatPercent(Number(label), 0)}`,
});

const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];

export function PrCurveChart({ champion, baseline, operatingPoints }: PrCurveChartProps) {
  const c = resamplePrCurve(champion);
  const b = resamplePrCurve(baseline);
  const data = c.map((precision, i) => ({ recall: i / (c.length - 1), champion: precision, baseline: b[i] }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 16, right: 20, bottom: 16, left: 0 }}>
        <CartesianGrid {...gridProps} />
        <XAxis
          type="number"
          dataKey="recall"
          domain={[0, 1]}
          ticks={ticks}
          tickFormatter={(v: number) => formatPercent(v, 0)}
          label={{ value: 'Recall', position: 'insideBottom', offset: -10, fill: CHART.tick, fontSize: 11 }}
          {...axisProps}
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          ticks={ticks}
          tickFormatter={(v: number) => formatPercent(v, 0)}
          width={64}
          label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 4, fill: CHART.tick, fontSize: 11 }}
          {...axisProps}
          axisLine={false}
        />
        <Tooltip content={tooltip} cursor={{ stroke: CHART.axis }} />
        <Legend verticalAlign="top" align="right" height={28} {...legendProps} />
        <Line type="linear" dataKey="baseline" name="Baseline (logistic regression)" stroke={SERIES.secondary} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
        <Line type="linear" dataKey="champion" name="Champion (gradient boosting)" stroke={SERIES.primary} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
        {operatingPoints.map((point, i) => (
          <ReferenceDot
            key={point.label}
            x={point.recall}
            y={point.precision}
            r={5}
            fill={CHART.ink}
            stroke={CHART.surface}
            strokeWidth={2}
            label={{ value: point.label, position: i === 0 ? 'right' : 'bottom', fill: CHART.ink, fontSize: 11, fontWeight: 600, offset: 10 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
