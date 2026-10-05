import { StatusBadge } from '@/components/common/status-badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { featureLabel } from '@/lib/features';
import { formatDate, formatPercent, formatStat } from '@/lib/format';
import type { DateRange, DriftRow, MonitoringReport } from '@/types';

interface DriftTableProps {
  rows: DriftRow[];
  thresholds: MonitoringReport['thresholds'];
  reference: DateRange;
  monthLabel: string;
}

/** Input drift per monitored feature against the training reference window. */
export function DriftTable({ rows, thresholds, reference, monthLabel }: DriftTableProps) {
  const sorted = [...rows].sort((a, b) => b.psi - a.psi);
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Data drift</CardTitle>
        <CardDescription>
          {monthLabel} inputs against the reference window ({formatDate(reference[0])} to {formatDate(reference[1])}), largest PSI first.
          PSI: warning from {thresholds.psi_warn}, alert from {thresholds.psi_alert}. KS: warning from {thresholds.ks_warn}, alert from{' '}
          {thresholds.ks_alert}; numeric features only.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Feature</TableHead>
              <TableHead numeric>PSI</TableHead>
              <TableHead numeric>KS</TableHead>
              <TableHead numeric>Missing (reference)</TableHead>
              <TableHead numeric>Missing (month)</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((row) => (
              <TableRow key={row.feature}>
                <TableCell>
                  <span className="block font-medium">{featureLabel(row.feature)}</span>
                  <span className="font-mono text-xs text-subtle">
                    {row.feature} · {row.kind}
                  </span>
                </TableCell>
                <TableCell numeric>{formatStat(row.psi)}</TableCell>
                <TableCell numeric>{row.ks === null ? <span title="Not computed for categorical features">n/a</span> : formatStat(row.ks)}</TableCell>
                <TableCell numeric className="text-muted-foreground">
                  {formatPercent(row.missing_ref)}
                </TableCell>
                <TableCell numeric>{formatPercent(row.missing_cur)}</TableCell>
                <TableCell>
                  <StatusBadge status={row.status} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
