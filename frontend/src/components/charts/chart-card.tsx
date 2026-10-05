'use client';

import { BarChart3, Table2 } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';

export interface ChartTable {
  columns: string[];
  rows: string[][];
}

interface ChartCardProps {
  title: string;
  description?: ReactNode;
  /** Chart height in px. */
  height?: number;
  /** The same data as a table, for screen readers and exact values. */
  table?: ChartTable;
  footer?: ReactNode;
  className?: string;
  children: ReactNode;
}

/** Card wrapper for a chart, with a chart/table toggle so no value is reachable by hover only. */
export function ChartCard({ title, description, height = 260, table, footer, className, children }: ChartCardProps) {
  const [showTable, setShowTable] = useState(false);
  const titleId = useId();

  return (
    <Card aria-labelledby={titleId} className={cn('flex min-w-0 flex-col', className)}>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <CardTitle id={titleId}>{title}</CardTitle>
          {description ? <CardDescription>{description}</CardDescription> : null}
        </div>
        {table ? (
          <Button variant="ghost" size="sm" className="shrink-0 text-muted-foreground" aria-pressed={showTable} onClick={() => setShowTable((v) => !v)}>
            {showTable ? <BarChart3 aria-hidden="true" /> : <Table2 aria-hidden="true" />}
            {showTable ? 'Chart' : 'Table'}
          </Button>
        ) : null}
      </CardHeader>
      <CardContent className="flex-1 pt-4">
        {showTable && table ? (
          <div className="overflow-y-auto" style={{ maxHeight: Math.max(height, 260) }}>
            <Table>
              <TableHeader>
                <TableRow>
                  {table.columns.map((column, i) => (
                    <TableHead key={column} numeric={i > 0}>
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {table.rows.map((row) => (
                  <TableRow key={row[0]}>
                    {row.map((cell, i) => (
                      <TableCell key={i} numeric={i > 0}>
                        {cell}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <div role="img" aria-label={`${title} chart. Use the Table button for the values.`} style={{ height }} className="w-full min-w-0">
            {children}
          </div>
        )}
        {footer ? <div className="mt-3 text-xs leading-relaxed text-muted-foreground">{footer}</div> : null}
      </CardContent>
    </Card>
  );
}
