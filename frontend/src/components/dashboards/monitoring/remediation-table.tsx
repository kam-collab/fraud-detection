import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatDate, formatINRCompact, formatNumber, formatPercent, formatStat } from '@/lib/format';
import type { MonitoringReport, RemediationRow } from '@/types';

const METRICS: { label: string; note?: string; value: (r: RemediationRow) => string }[] = [
  { label: 'PR-AUC', value: (r) => formatStat(r.pr_auc) },
  { label: 'ROC-AUC', value: (r) => formatStat(r.roc_auc) },
  { label: 'Precision', value: (r) => formatPercent(r.precision) },
  { label: 'Recall', value: (r) => formatPercent(r.recall) },
  { label: 'F1', value: (r) => formatStat(r.f1) },
  { label: 'False positives', note: 'genuine flagged', value: (r) => formatNumber(r.fp) },
  { label: 'False negatives', note: 'fraud missed', value: (r) => formatNumber(r.fn) },
  { label: 'False declines', note: 'genuine blocked', value: (r) => formatNumber(r.false_declines) },
  { label: 'Review queue', value: (r) => formatNumber(r.review_queue) },
  { label: 'Fraud loss', value: (r) => formatINRCompact(r.fraud_loss_amount) },
  { label: 'Total cost', value: (r) => formatINRCompact(r.total_cost) },
  { label: 'Review threshold', value: (r) => formatPercent(r.review_threshold, 1) },
  { label: 'Block threshold', value: (r) => formatPercent(r.block_threshold, 1) },
];

/** September re-scored three ways: do nothing, re-tune thresholds, or promote the challenger. */
export function RemediationTable({ remediation }: { remediation: MonitoringReport['remediation'] }) {
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>September remediation</CardTitle>
        <CardDescription>
          The same window ({formatDate(remediation.window[0])} to {formatDate(remediation.window[1])}) scored under each option, one
          option per column.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Metric</TableHead>
              {remediation.rows.map((row) => (
                <TableHead key={row.model} numeric className="min-w-40 whitespace-normal text-foreground">
                  {row.model}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {METRICS.map((metric) => (
              <TableRow key={metric.label}>
                <TableCell className="whitespace-nowrap">
                  {metric.label}
                  {metric.note ? <span className="ml-1.5 text-xs text-subtle">{metric.note}</span> : null}
                </TableCell>
                {remediation.rows.map((row) => (
                  <TableCell key={row.model} numeric>
                    {metric.value(row)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
