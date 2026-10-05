'use client';

import { CheckCircle2, Loader2, ThumbsDown, ThumbsUp, XCircle } from 'lucide-react';
import { ErrorState } from '@/components/common/error-state';
import { ExplanationCard } from '@/components/common/explanation-card';
import { ReasonsTable } from '@/components/common/reasons-table';
import { ScoreMeter } from '@/components/common/score-meter';
import { BandBadge } from '@/components/common/status-badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useExplanation, useReviewDecision } from '@/hooks';
import { formatDateTime, formatINR, formatPercent } from '@/lib/format';
import type { ReviewDecisionValue, ReviewItem } from '@/types';
import { decisionName, isCorrect, labelName } from './outcome';

interface DetailPanelProps {
  item: ReviewItem;
  thresholds?: { review: number; block: number };
}

function Outcome({ decision, actualLabel }: { decision: ReviewDecisionValue; actualLabel: 0 | 1 }) {
  const correct = isCorrect(decision, actualLabel);
  const Icon = correct ? CheckCircle2 : XCircle;
  return (
    <div role="status" className={`rounded-lg border p-3 text-sm ${correct ? 'border-ok-fg/30 bg-ok-bg' : 'border-alert-fg/30 bg-alert-bg'}`}>
      <p className={`flex items-center gap-1.5 font-semibold ${correct ? 'text-ok-fg' : 'text-alert-fg'}`}>
        <Icon className="size-4" aria-hidden="true" />
        {correct ? 'Correct decision' : 'Wrong decision'}
      </p>
      <p className="mt-1">
        {decisionName(decision)}; the transaction was actually <strong>{labelName(actualLabel)}</strong>.{' '}
        {correct
          ? null
          : actualLabel === 1
            ? 'Approving it would have let a fraud through.'
            : 'Declining it would have been a false decline for a genuine customer.'}
      </p>
    </div>
  );
}

/** Reasons, explanation and the approve/decline action for one queued transaction. */
export function DetailPanel({ item, thresholds }: DetailPanelProps) {
  const explain = useExplanation(item.request_id);
  const decide = useReviewDecision();

  // The mutation answers first; the refreshed queue row carries the same facts afterwards.
  const fresh = decide.data?.request_id === item.request_id ? decide.data : null;
  const decision = fresh?.decision ?? item.decision;
  const actualLabel = fresh?.actual_label ?? item.actual_label;

  const facts: [string, string][] = [
    ['Amount', formatINR(item.amount, { decimals: true })],
    ['Time', formatDateTime(item.request_time)],
    ['Payment method', item.service_type.toUpperCase()],
    ['Issuer bank', item.issuer_bank ?? '–'],
    ['Merchant category', `${item.mcc_title ?? '–'} (${item.mcc_code})`],
    ['Merchant risk tier', item.merchant_type],
    ['Location', [item.merchant_city, item.merchant_state].filter(Boolean).join(', ') || '–'],
  ];

  return (
    <Card aria-label={`Transaction ${item.request_id}`}>
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
        <CardTitle className="font-mono">{item.request_id}</CardTitle>
        <BandBadge band={item.band} />
      </CardHeader>
      <CardContent className="space-y-5">
        <div>
          <p className="text-xs font-medium text-muted-foreground">Fraud score</p>
          <p className="text-3xl font-semibold tracking-tight">{formatPercent(item.score, 1)}</p>
          {thresholds ? (
            <div className="mt-2">
              <ScoreMeter score={item.score} thresholds={thresholds} />
            </div>
          ) : null}
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          {facts.map(([label, value]) => (
            <div key={label} className="min-w-0">
              <dt className="text-xs text-muted-foreground">{label}</dt>
              <dd className="break-words font-medium">{value}</dd>
            </div>
          ))}
        </dl>

        {explain.isPending ? (
          <div role="status" className="space-y-2">
            <span className="sr-only">Loading reasons…</span>
            <Skeleton className="h-40" />
            <Skeleton className="h-16" />
          </div>
        ) : explain.isError ? (
          <ErrorState error={explain.error} onRetry={() => explain.refetch()} title="Could not load the reasons" />
        ) : (
          <>
            <div>
              <h3 className="mb-1 text-sm font-semibold">Top reasons</h3>
              <div className="-mx-3">
                <ReasonsTable reasons={explain.data.reasons} />
              </div>
            </div>
            <ExplanationCard key={item.request_id} requestId={item.request_id} explanation={explain.data.explanation} />
          </>
        )}

        <div className="space-y-3 border-t pt-4">
          {decision && actualLabel !== null ? (
            <Outcome decision={decision} actualLabel={actualLabel} />
          ) : (
            <>
              <p className="text-sm text-muted-foreground">Your call. The true label is revealed after you decide.</p>
              <div className="flex flex-wrap gap-2">
                <Button variant="approve" disabled={decide.isPending} onClick={() => decide.mutate({ requestId: item.request_id, decision: 'approve' })}>
                  {decide.isPending && decide.variables?.decision === 'approve' ? <Loader2 className="animate-spin" aria-hidden="true" /> : <ThumbsUp aria-hidden="true" />}
                  Approve
                </Button>
                <Button variant="decline" disabled={decide.isPending} onClick={() => decide.mutate({ requestId: item.request_id, decision: 'decline' })}>
                  {decide.isPending && decide.variables?.decision === 'decline' ? <Loader2 className="animate-spin" aria-hidden="true" /> : <ThumbsDown aria-hidden="true" />}
                  Decline
                </Button>
              </div>
              {decide.isError ? <ErrorState error={decide.error} title="The decision was not saved" /> : null}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
