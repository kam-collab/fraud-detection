import { CheckCircle2, Clock, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import type { ReviewDecisionValue } from '@/types';

export const isCorrect = (decision: ReviewDecisionValue, actualLabel: 0 | 1) => (decision === 'decline') === (actualLabel === 1);
export const labelName = (actualLabel: 0 | 1) => (actualLabel === 1 ? 'fraud' : 'genuine');
export const decisionName = (decision: ReviewDecisionValue) => (decision === 'approve' ? 'Approved' : 'Declined');

/** Queue-table badge: pending, or the analyst's decision and whether it matched the true label. */
export function OutcomeBadge({ decision, actualLabel }: { decision: ReviewDecisionValue | null; actualLabel: 0 | 1 | null }) {
  if (!decision) {
    return (
      <Badge variant="neutral">
        <Clock aria-hidden="true" />
        Pending
      </Badge>
    );
  }
  if (actualLabel === null) return <Badge variant="outline">{decisionName(decision)}</Badge>;
  const correct = isCorrect(decision, actualLabel);
  const Icon = correct ? CheckCircle2 : XCircle;
  return (
    <Badge variant={correct ? 'ok' : 'alert'}>
      <Icon aria-hidden="true" />
      {decisionName(decision)} · {correct ? 'correct' : 'wrong'}
    </Badge>
  );
}
