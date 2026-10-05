'use client';

import { MetricTrendChart, type TrendPoint } from '@/components/charts';
import { StatusBadge } from '@/components/common/status-badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDate, formatMonth, formatNumber, formatPercent, formatStat } from '@/lib/format';
import type { Interval, MonitoringMonth, MonitoringReport, Status } from '@/types';
import { MonthCompareTable, type CompareRow } from './month-compare-table';

type MetricKey = 'pr_auc' | 'precision' | 'recall';

const METRICS: { key: MetricKey; name: string }[] = [
  { key: 'pr_auc', name: 'PR-AUC' },
  { key: 'precision', name: 'Precision' },
  { key: 'recall', name: 'Recall' },
];

const short = (month: string) => formatMonth(month).slice(0, 3);

function WithStatus({ value, status }: { value: string; status: Status }) {
  return (
    <span className="inline-flex items-center justify-end gap-2">
      {value}
      <StatusBadge status={status} />
    </span>
  );
}

interface PerformanceSectionProps {
  months: MonitoringMonth[];
  baseline: MonitoringReport['baseline'];
  selected: string;
}

/** PR-AUC, precision and recall per month against the release baseline and its 95% CI. */
export function PerformanceSection({ months, baseline, selected }: PerformanceSectionProps) {
  const current = months.find((m) => m.month === selected);
  const interval = (ci: Interval) => `${formatStat(ci[0], 2)} to ${formatStat(ci[1], 2)}`;

  const rows: CompareRow[] = [
    ...METRICS.map(
      ({ key, name }): CompareRow => ({
        label: name,
        baseline: (
          <>
            {formatStat(baseline.metrics[key])}
            <span className="block text-xs text-subtle">CI {interval(baseline.ci95[key])}</span>
          </>
        ),
        value: (m) => <WithStatus value={formatStat(m.performance[key])} status={m.performance[`${key}_status`]} />,
      })
    ),
    { label: 'F1', baseline: formatStat(baseline.metrics.f1), value: (m) => <WithStatus value={formatStat(m.performance.f1)} status={m.performance.f1_status} /> },
    { label: 'ROC-AUC', baseline: formatStat(baseline.metrics.roc_auc), value: (m) => formatStat(m.performance.roc_auc) },
    { label: 'False positive rate', baseline: formatPercent(baseline.metrics.false_positive_rate, 2), value: (m) => formatPercent(m.performance.false_positive_rate, 2) },
    { label: 'False positives', note: 'genuine flagged', baseline: formatNumber(baseline.metrics.fp), value: (m) => formatNumber(m.performance.fp) },
    { label: 'False negatives', note: 'fraud missed', baseline: formatNumber(baseline.metrics.fn), value: (m) => formatNumber(m.performance.fn) },
    { label: 'Fraud rate', baseline: formatPercent(baseline.metrics.fraud_rate, 2), value: (m) => formatPercent(m.performance.fraud_rate, 2) },
  ];

  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Performance trend</CardTitle>
        <CardDescription>
          At the review threshold. The grey line is the June release baseline and the band its 95% confidence interval; a point below the
          band is a real drop, not noise.
          {current ? ` Labels for ${formatMonth(current.month)} are complete from ${formatDate(current.performance.labels_available_from)}.` : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 md:grid-cols-3">
          {METRICS.map(({ key, name }) => {
            const data: TrendPoint[] = [{ label: 'Jun', value: baseline.metrics[key] }, ...months.map((m) => ({ label: short(m.month), value: m.performance[key] }))];
            return (
              <figure key={key} className="min-w-0">
                <figcaption className="mb-1 text-xs font-medium text-muted-foreground">{name}</figcaption>
                <div role="img" aria-label={`${name} by month: ${data.map((d) => `${d.label} ${formatStat(d.value, 2)}`).join(', ')}`} className="h-44">
                  <MetricTrendChart name={name} data={data} baseline={baseline.metrics[key]} ci={baseline.ci95[key]} highlight={short(selected)} />
                </div>
              </figure>
            );
          })}
        </div>
        <div className="-mx-3">
          <MonthCompareTable rows={rows} months={months} selected={selected} />
        </div>
      </CardContent>
    </Card>
  );
}
