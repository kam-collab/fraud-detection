'use client';

import { useId, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Segmented } from '@/components/ui/segmented';
import { formatNumber, formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import type { ModelEvaluation } from '@/types';

type View = 'review' | 'block';

interface CellProps {
  name: string;
  meaning: string;
  count: number;
  total: number;
  tone: 'good' | 'bad';
}

function Cell({ name, meaning, count, total, tone }: CellProps) {
  return (
    <td className={cn('rounded-lg border p-3 align-top', tone === 'good' ? 'bg-accent' : 'bg-muted')}>
      <p className="text-xs font-medium text-muted-foreground">{name}</p>
      <p className="tabular mt-1 text-xl font-semibold">{formatNumber(count)}</p>
      <p className="mt-0.5 text-xs text-subtle">
        {meaning} · {formatPercent(count / total, 2)}
      </p>
    </td>
  );
}

/** Confusion matrix on the hold-out at either decision threshold. */
export function ConfusionMatrix({ evaluation }: { evaluation: ModelEvaluation }) {
  const [view, setView] = useState<View>('review');
  const titleId = useId();
  const m = view === 'review' ? evaluation.test : evaluation.test_block_band;
  const total = m.tp + m.fp + m.fn + m.tn;
  const flagged = view === 'review' ? 'Flagged (review or block)' : 'Blocked';
  const passed = view === 'review' ? 'Approved' : 'Not blocked';

  return (
    <Card aria-labelledby={titleId} className="min-w-0">
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <CardTitle id={titleId}>Confusion matrix</CardTitle>
          <CardDescription>
            June hold-out, {formatNumber(total)} transactions, threshold {formatPercent(m.threshold, 1)}.
          </CardDescription>
        </div>
        <Segmented<View>
          label="Decision threshold"
          value={view}
          onChange={setView}
          options={[
            { value: 'review', label: 'Review threshold' },
            { value: 'block', label: 'Block threshold' },
          ]}
        />
      </CardHeader>
      <CardContent>
        <div className="relative overflow-x-auto">
          <table className="w-full min-w-[26rem] border-separate border-spacing-2 text-sm">
            <thead>
              <tr>
                <td />
                <th scope="col" className="px-1 pb-1 text-left text-xs font-medium text-muted-foreground">
                  {flagged}
                </th>
                <th scope="col" className="px-1 pb-1 text-left text-xs font-medium text-muted-foreground">
                  {passed}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row" className="w-24 pr-1 text-left text-xs font-medium text-muted-foreground">
                  Actual fraud
                </th>
                <Cell name="True positive" meaning="Fraud caught" count={m.tp} total={total} tone="good" />
                <Cell name="False negative" meaning="Fraud missed" count={m.fn} total={total} tone="bad" />
              </tr>
              <tr>
                <th scope="row" className="pr-1 text-left text-xs font-medium text-muted-foreground">
                  Actual genuine
                </th>
                <Cell name="False positive" meaning={view === 'block' ? 'False decline' : 'Genuine flagged'} count={m.fp} total={total} tone="bad" />
                <Cell name="True negative" meaning="Genuine passed" count={m.tn} total={total} tone="good" />
              </tr>
            </tbody>
          </table>
        </div>
        <dl className="mt-3 grid grid-cols-3 gap-3 text-sm">
          {[
            ['Precision', formatPercent(m.precision)],
            ['Recall', formatPercent(m.recall)],
            ['False positive rate', formatPercent(m.false_positive_rate, 2)],
          ].map(([label, value]) => (
            <div key={label}>
              <dt className="text-xs text-muted-foreground">{label}</dt>
              <dd className="tabular font-semibold">{value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}
