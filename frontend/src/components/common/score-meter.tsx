import { formatPercent } from '@/lib/format';

interface ScoreMeterProps {
  score: number;
  thresholds: { review: number; block: number };
}

/** Fraud score on a 0-100% track with the review and block thresholds marked. */
export function ScoreMeter({ score, thresholds }: ScoreMeterProps) {
  const pct = (v: number) => `${Math.min(100, Math.max(0, v * 100))}%`;
  const fill = score >= thresholds.block ? 'bg-alert' : score >= thresholds.review ? 'bg-warn' : 'bg-ok';

  return (
    <div className="space-y-1.5">
      <div
        role="meter"
        aria-label="Fraud score"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number((score * 100).toFixed(1))}
        aria-valuetext={`${formatPercent(score)}; review from ${formatPercent(thresholds.review)}, block from ${formatPercent(thresholds.block)}`}
        className="relative h-3 rounded-full bg-muted"
      >
        <div className={`h-full rounded-full ${fill}`} style={{ width: pct(score), minWidth: '0.375rem' }} />
        {[thresholds.review, thresholds.block].map((t) => (
          <span key={t} aria-hidden="true" className="absolute -top-1 h-5 w-0.5 bg-foreground" style={{ left: pct(t) }} />
        ))}
      </div>
      <div className="relative h-4 text-[11px] text-muted-foreground" aria-hidden="true">
        <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: pct(thresholds.review) }}>
          Review {formatPercent(thresholds.review)}
        </span>
        <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: pct(thresholds.block) }}>
          Block {formatPercent(thresholds.block)}
        </span>
      </div>
    </div>
  );
}
