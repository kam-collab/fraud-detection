/** Shared Recharts styling. Colours are CSS variables so charts follow the light/dark theme. */

export const SERIES = {
  primary: 'var(--series-1)',
  secondary: 'var(--series-2)',
  tertiary: 'var(--series-3)',
  neutral: 'var(--series-neutral)',
} as const;

export const CHART = {
  grid: 'var(--chart-grid)',
  axis: 'var(--chart-axis)',
  tick: 'var(--chart-tick)',
  surface: 'var(--card)',
  ink: 'var(--foreground)',
} as const;

export const axisProps = {
  tick: { fill: CHART.tick, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: CHART.axis },
} as const;

export const gridProps = {
  stroke: CHART.grid,
  strokeDasharray: undefined,
} as const;

export const legendProps = {
  iconSize: 10,
  wrapperStyle: { fontSize: 12, color: 'var(--muted-foreground)' },
} as const;
