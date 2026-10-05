'use client';

import { useQuery } from '@tanstack/react-query';
import { fraudApi } from '@/lib/api';
import { fraudKeys } from './query-keys';

/**
 * Reasons and explanation text for a scored transaction.
 * `llm: true` asks the API for the LLM-written note; the API falls back to the
 * template when no LLM is configured, which is visible in `explanation.source`.
 */
export function useExplanation(requestId: string | null, { llm = false, enabled = true }: { llm?: boolean; enabled?: boolean } = {}) {
  return useQuery({
    queryKey: fraudKeys.explain(requestId ?? '', llm),
    queryFn: ({ signal }) => fraudApi.explain(requestId as string, llm, signal),
    enabled: enabled && !!requestId,
    staleTime: Infinity,
  });
}
