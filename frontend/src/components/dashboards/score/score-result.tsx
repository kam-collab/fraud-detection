import { AlertTriangle } from 'lucide-react';
import { ExplanationCard } from '@/components/common/explanation-card';
import { ReasonsTable } from '@/components/common/reasons-table';
import { ScoreMeter } from '@/components/common/score-meter';
import { BandBadge } from '@/components/common/status-badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatPercent } from '@/lib/format';
import type { ScoreResult as ScoreResultType } from '@/types';

export function ScoreResult({ result }: { result: ScoreResultType }) {
  return (
    <div className="space-y-4" aria-live="polite">
      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle>Decision</CardTitle>
          <span className="font-mono text-xs text-subtle">
            {result.request_id} · model v{result.model_version}
          </span>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div>
              <p className="text-xs font-medium text-muted-foreground">Fraud score</p>
              <p className="text-4xl font-semibold tracking-tight">{formatPercent(result.score, result.score < 0.1 ? 2 : 1)}</p>
            </div>
            <BandBadge band={result.band} className="mb-1.5 px-3 py-1 text-sm" />
          </div>
          <ScoreMeter score={result.score} thresholds={result.thresholds} />
          <dl className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <dt className="text-xs text-muted-foreground">Review threshold</dt>
              <dd className="tabular font-medium">{formatPercent(result.thresholds.review, 2)}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Block threshold</dt>
              <dd className="tabular font-medium">{formatPercent(result.thresholds.block, 2)}</dd>
            </div>
          </dl>
          {result.warnings.length > 0 ? (
            <div className="rounded-lg border border-warn-fg/30 bg-warn-bg p-3 text-sm text-warn-fg">
              <p className="flex items-center gap-1.5 font-semibold">
                <AlertTriangle className="size-4" aria-hidden="true" />
                Data warnings
              </p>
              <ul className="mt-1 list-disc space-y-0.5 pl-5 text-foreground">
                {result.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Top reasons</CardTitle>
        </CardHeader>
        <CardContent className="px-2 pt-3">
          <ReasonsTable reasons={result.reasons} />
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          {/* Keyed so a new score resets any LLM note requested for the previous one. */}
          <ExplanationCard key={`${result.request_id}-${result.score}`} requestId={result.request_id} explanation={result.explanation} />
        </CardContent>
      </Card>
    </div>
  );
}
