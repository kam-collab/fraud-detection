'use client';

import { Loader2, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { SourceBadge } from '@/components/common/status-badge';
import { Button } from '@/components/ui/button';
import { useExplanation } from '@/hooks';
import type { Explanation } from '@/types';

interface ExplanationCardProps {
  requestId: string;
  /** Explanation already returned with the score / explain call (template unless the API says otherwise). */
  explanation: Explanation;
}

/**
 * Shows the explanation text with its source. The LLM-written note is only
 * requested when the analyst asks for it. The server ignores the request unless
 * the operator has enabled EXPLANATIONS_LLM, and then returns the template;
 * the card says so rather than hiding it.
 */
export function ExplanationCard({ requestId, explanation }: ExplanationCardProps) {
  const [wantLlm, setWantLlm] = useState(false);
  const llm = useExplanation(requestId, { llm: true, enabled: wantLlm });

  const shown = wantLlm && llm.data ? llm.data.explanation : explanation;
  const fellBack = wantLlm && llm.data && llm.data.explanation.source !== 'llm';

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">Explanation</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Source</span>
          <SourceBadge source={shown.source} model={shown.model} />
        </div>
      </div>
      <p className="text-sm leading-relaxed">{shown.text}</p>
      {shown.source !== 'llm' ? (
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" size="sm" onClick={() => (wantLlm ? llm.refetch() : setWantLlm(true))} disabled={llm.isFetching}>
            {llm.isFetching ? <Loader2 className="animate-spin" aria-hidden="true" /> : <Sparkles aria-hidden="true" />}
            Ask for LLM-written note
          </Button>
          <p role="status" className="text-xs text-muted-foreground">
            {fellBack ? 'The API returned the template: LLM notes are not enabled on this server (EXPLANATIONS_LLM).' : null}
            {wantLlm && llm.isError ? 'The LLM note could not be fetched; the template is shown.' : null}
          </p>
        </div>
      ) : null}
    </div>
  );
}
