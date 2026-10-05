'use client';

import { ChartCard, PatternRecallChart, type PatternDatum } from '@/components/charts';
import { patternLabel } from '@/lib/features';
import { formatNumber, formatPercent } from '@/lib/format';
import type { PatternRecall } from '@/types';

interface PatternSectionProps {
  current: PatternRecall[];
  /** Recall by pattern on the June hold-out, when the model report is available. */
  baseline?: PatternRecall[];
  monthLabel: string;
}

export function PatternSection({ current, baseline, monthLabel }: PatternSectionProps) {
  const base = new Map((baseline ?? []).map((p) => [p.group, p]));
  const data: PatternDatum[] = current.map((p) => ({
    label: patternLabel(p.group),
    baseline: base.get(p.group)?.recall ?? null,
    current: p.recall,
    frauds: p.frauds,
  }));
  const unseen = current.filter((p) => baseline && !base.has(p.group)).map((p) => patternLabel(p.group));

  return (
    <ChartCard
      title="Recall by fraud pattern"
      description="Share of each pattern's frauds flagged for review or block. Patterns are tags from the synthetic generator."
      height={Math.max(220, data.length * 46)}
      table={{
        columns: ['Pattern', 'June baseline', monthLabel, 'Frauds this month'],
        rows: data.map((d) => [d.label, d.baseline === null ? 'n/a' : formatPercent(d.baseline), formatPercent(d.current), formatNumber(d.frauds)]),
      }}
      footer={
        <>
          {unseen.length > 0 ? <p>No June baseline for {unseen.join(', ')}: the pattern was not present in the hold-out window.</p> : null}
          {!baseline ? <p>The June baseline per pattern is unavailable (model report not loaded).</p> : null}
        </>
      }
    >
      <PatternRecallChart data={data} currentLabel={monthLabel} />
    </ChartCard>
  );
}
