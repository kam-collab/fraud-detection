import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { formatINRCompact, formatNumber, formatPercent } from '@/lib/format';
import type { DateRange, MonitoringMonth, MonitoringReport } from '@/types';
import { MonthCompareTable, type CompareRow } from './month-compare-table';

interface BusinessTableProps {
  months: MonitoringMonth[];
  baseline: MonitoringReport['baseline']['business'];
  /** Window the baseline was measured on. */
  baselineWindow: DateRange;
  selected: string;
}

function daysBetween([start, end]: DateRange): number {
  return Math.round((Date.parse(`${end}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86_400_000);
}

/** What the drift costs: approvals, analyst workload, false declines and fraud loss. */
export function BusinessTable({ months, baseline, baselineWindow, selected }: BusinessTableProps) {
  const days = daysBetween(baselineWindow);
  const rows: CompareRow[] = [
    { label: 'Transactions', baseline: formatNumber(baseline.transactions), value: (m) => formatNumber(m.business.transactions) },
    { label: 'Approval rate', note: 'approved without review', baseline: formatPercent(baseline.approval_rate, 2), value: (m) => formatPercent(m.business.approval_rate, 2) },
    {
      label: 'Review queue per day',
      baseline: formatNumber(baseline.review_queue_per_day, 1),
      value: (m) => formatNumber(m.business.review_queue_per_day, 1),
    },
    { label: 'Review rate', baseline: formatPercent(baseline.review_rate, 2), value: (m) => formatPercent(m.business.review_rate, 2) },
    { label: 'Genuine sent to review', baseline: formatNumber(baseline.genuine_sent_to_review), value: (m) => formatNumber(m.business.genuine_sent_to_review) },
    { label: 'False declines', note: 'genuine blocked', baseline: formatNumber(baseline.false_declines), value: (m) => formatNumber(m.business.false_declines) },
    { label: 'False-decline rate', baseline: formatPercent(baseline.false_decline_rate, 3), value: (m) => formatPercent(m.business.false_decline_rate, 3) },
    { label: 'False-decline amount', baseline: formatINRCompact(baseline.false_decline_amount), value: (m) => formatINRCompact(m.business.false_decline_amount) },
    { label: 'Fraud loss', note: 'fraud not stopped', baseline: formatINRCompact(baseline.fraud_loss_amount), value: (m) => formatINRCompact(m.business.fraud_loss_amount) },
    { label: 'Fraud value caught', baseline: formatPercent(baseline.fraud_value_recall), value: (m) => formatPercent(m.business.fraud_value_recall) },
    { label: 'Total cost', baseline: formatINRCompact(baseline.total_cost), value: (m) => formatINRCompact(m.business.total_cost) },
  ];

  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Business metrics</CardTitle>
        <CardDescription>
          The June baseline covers {days} days, so compare rates and per-day figures across columns rather than counts and amounts.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-3">
        <MonthCompareTable rows={rows} months={months} selected={selected} />
      </CardContent>
    </Card>
  );
}
