'use client';

import { ChartCard, ScoreHistogramChart, type HistogramBin } from '@/components/charts';
import { StatusBadge } from '@/components/common/status-badge';
import { formatPercent, formatPointDelta, formatStat } from '@/lib/format';
import type { PredictionDrift, ScoreHistogram } from '@/types';

interface PredictionDriftSectionProps {
  drift: PredictionDrift;
  histogram: ScoreHistogram;
  month: string;
  monthLabel: string;
}

export function toBins(histogram: ScoreHistogram, month: string): HistogramBin[] {
  const current = histogram[month] ?? [];
  return histogram.baseline_june.map((baseline, i) => ({
    bin: `${histogram.edges[i].toFixed(2)}–${histogram.edges[i + 1].toFixed(2)}`,
    baseline,
    current: current[i] ?? 0,
  }));
}

/** Score distribution against the June baseline, with the summary drift statistics. */
export function PredictionDriftSection({ drift, histogram, month, monthLabel }: PredictionDriftSectionProps) {
  const bins = toBins(histogram, month);
  const change = drift.flag_rate_change;
  const stats: [string, string, string][] = [
    ['Score PSI', formatStat(drift.score_psi), 'against the June score distribution'],
    [
      'Flagged for review or block',
      formatPercent(drift.flag_rate, 2),
      `${change > 0 ? '+' : change < 0 ? '−' : '±'}${formatPercent(Math.abs(change), 0)} vs baseline ${formatPercent(drift.baseline_flag_rate, 2)}`,
    ],
    ['Blocked', formatPercent(drift.block_rate, 2), `${formatPointDelta(drift.block_rate, drift.baseline_block_rate, 2)} vs baseline ${formatPercent(drift.baseline_block_rate, 2)}`],
    ['Mean score', formatPercent(drift.mean_score, 2), `baseline ${formatPercent(drift.baseline_mean_score, 2)}`],
  ];

  return (
    <ChartCard
      title="Score distribution vs June baseline"
      description="Share of transactions per fraud-score bin. The vertical axis is logarithmic because about 97% of payments score below 5%."
      height={280}
      table={{
        columns: ['Score bin', 'June baseline', monthLabel],
        rows: bins.map((b) => [b.bin, formatPercent(b.baseline, 3), formatPercent(b.current, 3)]),
      }}
      footer={
        <dl className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-5">
          <div>
            <dt className="text-xs text-muted-foreground">Prediction drift</dt>
            <dd className="mt-0.5">
              <StatusBadge status={drift.status} />
            </dd>
            <dd className="mt-0.5 text-xs text-subtle">From score PSI and the change in flag rate</dd>
          </div>
          {stats.map(([label, value, note]) => (
            <div key={label}>
              <dt className="text-xs text-muted-foreground">{label}</dt>
              <dd className="tabular font-semibold text-foreground">{value}</dd>
              <dd className="tabular text-xs text-subtle">{note}</dd>
            </div>
          ))}
        </dl>
      }
    >
      <ScoreHistogramChart data={bins} currentLabel={monthLabel} />
    </ChartCard>
  );
}
