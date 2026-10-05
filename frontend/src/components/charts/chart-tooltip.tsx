import type { ReactNode } from 'react';
import type { TooltipContentProps } from 'recharts';

type Row = Record<string, unknown>;

interface ChartTooltipOptions {
  /** Formats each series value. */
  formatValue: (value: number, dataKey: string) => string;
  /** Formats the heading (the x value). */
  formatLabel?: (label: string | number, row: Row) => string;
  /** Extra lines derived from the hovered data row (counts, sample sizes). */
  extra?: (row: Row) => Array<[string, string]>;
}

/** Tooltip in text tokens; the swatch beside each line carries the series identity. */
export function chartTooltip({ formatValue, formatLabel, extra }: ChartTooltipOptions) {
  function ChartTooltipContent({ active, payload, label }: TooltipContentProps): ReactNode {
    if (!active || !payload?.length) return null;
    const row = (payload[0]?.payload ?? {}) as Row;
    const heading = label === undefined ? '' : formatLabel ? formatLabel(label, row) : String(label);
    return (
      <div className="rounded-lg border bg-card px-3 py-2 text-xs shadow-md">
        {heading ? <p className="mb-1 font-semibold text-foreground">{heading}</p> : null}
        <ul className="space-y-0.5">
          {payload.map((entry) => (
            <li key={String(entry.dataKey)} className="flex items-center gap-2 text-muted-foreground">
              <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: entry.color ?? entry.stroke ?? entry.fill }} />
              <span>{entry.name}</span>
              <span className="tabular ml-auto pl-3 font-medium text-foreground">
                {typeof entry.value === 'number' ? formatValue(entry.value, String(entry.dataKey)) : '–'}
              </span>
            </li>
          ))}
          {extra?.(row).map(([name, value]) => (
            <li key={name} className="flex items-center gap-2 text-muted-foreground">
              <span>{name}</span>
              <span className="tabular ml-auto pl-3 text-foreground">{value}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  return ChartTooltipContent;
}
