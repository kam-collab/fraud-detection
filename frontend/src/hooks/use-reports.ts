'use client';

import { useQuery } from '@tanstack/react-query';
import { fraudApi } from '@/lib/api';
import { fraudKeys } from './query-keys';

/** EDA summaries for the development period. */
export function useOverview() {
  return useQuery({ queryKey: fraudKeys.overview(), queryFn: ({ signal }) => fraudApi.overview(signal) });
}

/** Model card and release evaluation on the June hold-out. */
export function useModelReport() {
  return useQuery({ queryKey: fraudKeys.model(), queryFn: ({ signal }) => fraudApi.model(signal) });
}

/** Drift and performance monitoring for the live months. */
export function useMonitoring() {
  return useQuery({ queryKey: fraudKeys.monitoring(), queryFn: ({ signal }) => fraudApi.monitoring(signal) });
}
