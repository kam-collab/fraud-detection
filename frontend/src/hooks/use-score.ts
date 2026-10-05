'use client';

import { useMutation } from '@tanstack/react-query';
import { fraudApi, type ApiError } from '@/lib/api';
import type { SampleKind, ScoreResult, Transaction } from '@/types';

export function useScoreTransaction() {
  return useMutation<ScoreResult, ApiError, Record<string, unknown>>({
    mutationFn: (transaction) => fraudApi.score(transaction),
  });
}

/** Fetches a stored transaction to pre-fill the scoring form. Each call returns a different one. */
export function useSampleTransaction() {
  return useMutation<Transaction, ApiError, SampleKind>({
    mutationFn: (kind) => fraudApi.sampleTransaction(kind),
  });
}
