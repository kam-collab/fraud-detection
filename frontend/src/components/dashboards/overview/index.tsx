'use client';

import { ChartCard, MonthlyFraudChart, PrCurveChart } from '@/components/charts';
import { ErrorState } from '@/components/common/error-state';
import { KpiTile } from '@/components/common/kpi-tile';
import { LoadingState } from '@/components/common/loading-state';
import { PageHeader, SectionHeading } from '@/components/common/page-header';
import { useModelReport, useOverview } from '@/hooks';
import { formatDate, formatINRCompact, formatMonth, formatNumber, formatPercent, formatStat } from '@/lib/format';
import type { Interval } from '@/types';
import { ConfusionMatrix } from './confusion-matrix';
import { EdaBreakdowns } from './eda-breakdowns';
import { ModelComparison, SelectionTable } from './model-comparison';

const ci = (interval: Interval, format: (v: number) => string) => `95% CI ${format(interval[0])} to ${format(interval[1])}`;

export function OverviewDashboard() {
  const overview = useOverview();
  const model = useModelReport();

  const header = (
    <PageHeader
      title="Overview"
      description="How the champion model performs on the June hold-out, and what fraud looks like in the development data. All data is synthetic."
    />
  );

  if (overview.isPending && model.isPending) {
    return (
      <>
        {header}
        <LoadingState label="Loading overview" tiles={6} blocks={2} />
      </>
    );
  }

  // One message instead of two when the whole API is down.
  if (overview.isError && model.isError) {
    return (
      <>
        {header}
        <ErrorState
          error={overview.error}
          onRetry={() => {
            overview.refetch();
            model.refetch();
          }}
        />
      </>
    );
  }

  const o = overview.data;
  const m = model.data;
  const champion = m?.champion;
  const test = champion?.test;
  const business = champion?.test_business;
  const firstLiveMonth = m ? m.windows.test[1].slice(0, 7) : undefined;

  return (
    <>
      {header}

      <section aria-label="Key figures" className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <KpiTile
          label="Transactions"
          value={o ? formatNumber(o.transactions) : '–'}
          hint={o ? `${formatDate(o.period[0])} to ${formatDate(o.period[1])}` : 'Development data unavailable'}
        />
        <KpiTile
          label="Fraud rate"
          value={o ? formatPercent(o.fraud_rate, 2) : '–'}
          hint={o ? `${formatNumber(o.frauds)} frauds · ${formatPercent(o.fraud_amount_share)} of amount` : undefined}
        />
        <KpiTile label="PR-AUC" value={test ? formatStat(test.pr_auc) : '–'} hint={champion ? ci(champion.test_ci95.pr_auc, formatStat) : 'June hold-out'} />
        <KpiTile
          label="Precision"
          value={test ? formatPercent(test.precision) : '–'}
          hint={champion ? ci(champion.test_ci95.precision, (v) => formatPercent(v)) : 'June hold-out'}
        />
        <KpiTile label="Recall" value={test ? formatPercent(test.recall) : '–'} hint={champion ? ci(champion.test_ci95.recall, (v) => formatPercent(v)) : 'June hold-out'} />
        <KpiTile
          label="False declines"
          value={business ? formatNumber(business.false_declines) : '–'}
          hint={
            business
              ? `${formatPercent(business.false_declines / business.transactions, 3)} of payments · ${formatINRCompact(business.false_decline_amount)}`
              : 'June hold-out'
          }
        />
      </section>
      <p className="text-xs text-subtle">
        Transactions and fraud rate cover the development period. PR-AUC, precision, recall (at the review threshold) and false declines
        (genuine payments blocked) are measured on the June hold-out
        {m ? ` (${formatDate(m.windows.test[0])} to ${formatDate(m.windows.test[1])}, ${formatNumber(m.rows.test.n)} transactions)` : ''}.
      </p>

      <SectionHeading
        title="Model quality on the June hold-out"
        description={
          champion
            ? `Three-band policy: approve below ${formatPercent(champion.review_threshold, 1)}, manual review from there, block from ${formatPercent(champion.block_threshold, 1)}.`
            : undefined
        }
      />
      {model.isPending ? (
        <LoadingState label="Loading model report" />
      ) : model.isError || !m || !champion ? (
        <ErrorState error={model.error} onRetry={() => model.refetch()} title="Could not load the model report" />
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-2">
            <ConfusionMatrix evaluation={champion} />
            <ChartCard
              title="Precision-recall curve"
              description="Champion against the baseline. The dots are the champion's review and block thresholds."
              height={300}
              table={{
                columns: ['Operating point', 'Threshold', 'Precision', 'Recall'],
                rows: [
                  ['Champion, review', formatPercent(champion.test.threshold, 1), formatPercent(champion.test.precision), formatPercent(champion.test.recall)],
                  ['Champion, block', formatPercent(champion.test_block_band.threshold, 1), formatPercent(champion.test_block_band.precision), formatPercent(champion.test_block_band.recall)],
                  ['Baseline, review', formatPercent(m.baseline.test.threshold, 1), formatPercent(m.baseline.test.precision), formatPercent(m.baseline.test.recall)],
                  ['Baseline, block', formatPercent(m.baseline.test_block_band.threshold, 1), formatPercent(m.baseline.test_block_band.precision), formatPercent(m.baseline.test_block_band.recall)],
                ],
              }}
            >
              <PrCurveChart
                champion={champion.test_pr_curve}
                baseline={m.baseline.test_pr_curve}
                operatingPoints={[
                  { label: 'Review', recall: champion.test.recall, precision: champion.test.precision },
                  { label: 'Block', recall: champion.test_block_band.recall, precision: champion.test_block_band.precision },
                ]}
              />
            </ChartCard>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <ModelComparison champion={champion} baseline={m.baseline} />
            <SelectionTable rows={m.selection_table} champion={champion.name} />
          </div>
        </>
      )}

      <SectionHeading
        title="Fraud in the data"
        description={o ? `Development period, ${formatDate(o.period[0])} to ${formatDate(o.period[1])}. Monthly fraud rate also covers the live months.` : undefined}
      />
      {overview.isPending ? (
        <LoadingState label="Loading data overview" />
      ) : overview.isError || !o ? (
        <ErrorState error={overview.error} onRetry={() => overview.refetch()} title="Could not load the data overview" />
      ) : (
        <>
          <ChartCard
            title="Fraud rate by month"
            description={firstLiveMonth ? `The model was released after June; ${formatMonth(firstLiveMonth)} onwards is live traffic.` : undefined}
            height={240}
            table={{
              columns: ['Month', 'Fraud rate', 'Transactions', 'Frauds'],
              rows: o.monthly.map((r) => [formatMonth(r.month), formatPercent(r.fraud_rate, 2), formatNumber(r.transactions), formatNumber(r.frauds)]),
            }}
          >
            <MonthlyFraudChart data={o.monthly} liveFrom={firstLiveMonth} />
          </ChartCard>
          <EdaBreakdowns overview={o} />
        </>
      )}
    </>
  );
}
