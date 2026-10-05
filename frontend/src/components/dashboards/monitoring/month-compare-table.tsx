import type { ReactNode } from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatMonth } from '@/lib/format';
import { cn } from '@/lib/utils';
import type { MonitoringMonth } from '@/types';

export interface CompareRow {
  label: string;
  note?: string;
  baseline: ReactNode;
  value: (month: MonitoringMonth) => ReactNode;
}

interface MonthCompareTableProps {
  rows: CompareRow[];
  months: MonitoringMonth[];
  selected: string;
  baselineLabel?: string;
}

/** Metrics as rows, the June baseline and each live month as columns; the selected month is marked. */
export function MonthCompareTable({ rows, months, selected, baselineLabel = 'June baseline' }: MonthCompareTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Metric</TableHead>
          <TableHead numeric>{baselineLabel}</TableHead>
          {months.map((m) => (
            <TableHead key={m.month} numeric className={cn(m.month === selected && 'bg-accent text-foreground')}>
              {formatMonth(m.month)}
              {m.month === selected ? <span className="sr-only"> (selected)</span> : null}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row) => (
          <TableRow key={row.label}>
            <TableCell>
              {row.label}
              {row.note ? <span className="ml-1.5 text-xs text-subtle">{row.note}</span> : null}
            </TableCell>
            <TableCell numeric className="text-muted-foreground">
              {row.baseline}
            </TableCell>
            {months.map((m) => (
              <TableCell key={m.month} numeric className={cn(m.month === selected && 'bg-accent font-medium')}>
                {row.value(m)}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
