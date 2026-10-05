import { ArrowDown, ArrowUp } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { featureLabel, formatFeatureValue } from '@/lib/features';
import type { Reason } from '@/types';

/** Top contributing features: this transaction's value against the typical value. */
export function ReasonsTable({ reasons }: { reasons: Reason[] }) {
  if (reasons.length === 0) return <p className="text-sm text-muted-foreground">No contributing features were returned.</p>;
  const max = Math.max(...reasons.map((r) => Math.abs(r.log_odds_contribution)), 0.001);

  return (
    <>
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Feature</TableHead>
          <TableHead numeric>Value</TableHead>
          <TableHead numeric>Typical</TableHead>
          <TableHead>Contribution</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {reasons.map((reason) => {
          const up = reason.log_odds_contribution >= 0;
          const Icon = up ? ArrowUp : ArrowDown;
          return (
            <TableRow key={reason.feature}>
              <TableCell>
                <span className="block font-medium">{featureLabel(reason.feature)}</span>
                <span className="break-all font-mono text-xs text-subtle">{reason.feature}</span>
              </TableCell>
              <TableCell numeric>{formatFeatureValue(reason.value)}</TableCell>
              <TableCell numeric className="text-muted-foreground">
                {formatFeatureValue(reason.typical)}
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <span className="tabular flex w-16 shrink-0 items-center gap-1 text-sm">
                    <Icon className="size-3.5 text-muted-foreground" aria-hidden="true" />
                    <span className="sr-only">{up ? 'Raises risk by' : 'Lowers risk by'}</span>
                    {up ? '+' : '−'}
                    {Math.abs(reason.log_odds_contribution).toFixed(2)}
                  </span>
                  <span className="h-2 min-w-6 flex-1 rounded-full bg-muted" aria-hidden="true">
                    <span
                      className={`block h-full rounded-full ${up ? 'bg-series-2' : 'bg-series-1'}`}
                      style={{ width: `${(Math.abs(reason.log_odds_contribution) / max) * 100}%` }}
                    />
                  </span>
                </div>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
    <p className="px-3 pt-2 text-xs text-subtle">Contribution is in log-odds: a positive value pushes the score towards fraud.</p>
    </>
  );
}
