'use client';

import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { fraudApi, type ApiError } from '@/lib/api';
import type { ReviewDecisionResult, ReviewDecisionValue } from '@/types';
import { fraudKeys } from './query-keys';

export function useReviewQueue(limit: number, offset: number) {
  return useQuery({
    queryKey: fraudKeys.reviewQueuePage(limit, offset),
    queryFn: ({ signal }) => fraudApi.reviewQueue(limit, offset, signal),
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });
}

export function useReviewDecision() {
  const queryClient = useQueryClient();
  return useMutation<ReviewDecisionResult, ApiError, { requestId: string; decision: ReviewDecisionValue }>({
    mutationFn: ({ requestId, decision }) => fraudApi.reviewDecision(requestId, decision),
    // The queue row now carries the decision and the revealed label.
    onSuccess: () => queryClient.invalidateQueries({ queryKey: fraudKeys.reviewQueue() }),
  });
}
