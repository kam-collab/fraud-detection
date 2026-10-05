import { Check } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatINRCompact, formatNumber, formatPercent, formatStat } from '@/lib/format';
import type { ModelEvaluation, SelectionRow } from '@/types';

interface Row {
  label: string;
  note?: string;
  value: (e: ModelEvaluation) => number;
  format: (v: number) => string;
  better: 'higher' | 'lower';
}

const pct = (v: number) => formatPercent(v);

const ROWS: Row[] = [
  { label: 'PR-AUC', value: (e) => e.test.pr_auc, format: formatStat, better: 'higher' },
  { label: 'ROC-AUC', value: (e) => e.test.roc_auc, format: formatStat, better: 'higher' },
  { label: 'Precision', note: 'review threshold', value: (e) => e.test.precision, format: pct, better: 'higher' },
  { label: 'Recall', note: 'review threshold', value: (e) => e.test.recall, format: pct, better: 'higher' },
  { label: 'F1', note: 'review threshold', value: (e) => e.test.f1, format: formatStat, better: 'higher' },
  { label: 'False positive rate', note: 'review threshold', value: (e) => e.test.false_positive_rate, format: (v) => formatPercent(v, 2), better: 'lower' },
  { label: 'Precision', note: 'block threshold', value: (e) => e.test_block_band.precision, format: pct, better: 'higher' },
  { label: 'Recall', note: 'block threshold', value: (e) => e.test_block_band.recall, format: pct, better: 'higher' },
  { label: 'False declines', note: 'genuine payments blocked', value: (e) => e.test_business.false_declines, format: formatNumber, better: 'lower' },
  { label: 'False-decline amount', value: (e) => e.test_business.false_decline_amount, format: formatINRCompact, better: 'lower' },
  { label: 'Fraud value caught', note: 'share of fraud amount', value: (e) => e.test_business.fraud_value_recall, format: pct, better: 'higher' },
  { label: 'Fraud loss', value: (e) => e.test_business.fraud_loss_amount, format: formatINRCompact, better: 'lower' },
  { label: 'Total cost', value: (e) => e.test_business.total_cost, format: formatINRCompact, better: 'lower' },
];

function Value({ text, wins }: { text: string; wins: boolean }) {
  return (
    <span className="inline-flex items-center justify-end gap-1.5">
      {wins ? (
        <>
          <Check className="size-3.5 text-ok-fg" aria-hidden="true" />
          <span className="sr-only">Better: </span>
        </>
      ) : null}
      <span className={wins ? 'font-semibold' : undefined}>{text}</span>
    </span>
  );
}

/** Champion vs baseline on the same hold-out; a tick marks the better value on each row. */
export function ModelComparison({ champion, baseline }: { champion: ModelEvaluation; baseline: ModelEvaluation }) {
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Champion vs baseline</CardTitle>
        <CardDescription>Both evaluated on the June hold-out, each at its own thresholds. A tick marks the better value.</CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Metric</TableHead>
              <TableHead numeric>
                Champion
                <span className="block font-normal text-subtle">{champion.name}</span>
              </TableHead>
              <TableHead numeric>
                Baseline
                <span className="block font-normal text-subtle">{baseline.name}</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {ROWS.map((row) => {
              const c = row.value(champion);
              const b = row.value(baseline);
              const championWins = row.better === 'higher' ? c > b : c < b;
              const baselineWins = row.better === 'higher' ? b > c : b < c;
              return (
                <TableRow key={`${row.label}-${row.note ?? ''}`}>
                  <TableCell>
                    {row.label}
                    {row.note ? <span className="ml-1.5 text-xs text-subtle">{row.note}</span> : null}
                  </TableCell>
                  <TableCell numeric>
                    <Value text={row.format(c)} wins={championWins} />
                  </TableCell>
                  <TableCell numeric>
                    <Value text={row.format(b)} wins={baselineWins} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

/** Candidates ranked on the validation window; the first row is the champion. */
export function SelectionTable({ rows, champion }: { rows: SelectionRow[]; champion: string }) {
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Model selection</CardTitle>
        <CardDescription>Candidates compared on the validation window (before calibration).</CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Candidate</TableHead>
              <TableHead numeric>PR-AUC</TableHead>
              <TableHead numeric>ROC-AUC</TableHead>
              <TableHead numeric>Brier</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.model}>
                <TableCell className={row.model === champion ? 'font-semibold' : undefined}>
                  {row.model}
                  {row.model === champion ? <span className="ml-1.5 text-xs font-normal text-subtle">champion</span> : null}
                </TableCell>
                <TableCell numeric>{formatStat(row.valid_pr_auc)}</TableCell>
                <TableCell numeric>{formatStat(row.valid_roc_auc)}</TableCell>
                <TableCell numeric>{formatStat(row.valid_brier_uncalibrated, 4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
