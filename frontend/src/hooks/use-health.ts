'use client';

import { useQuery } from '@tanstack/react-query';
import { fraudApi } from '@/lib/api';
import { fraudKeys } from './query-keys';

/** Polls /readiness so the header can show whether the API and model are up. */
export function useReadiness() {
  return useQuery({
    queryKey: fraudKeys.readiness(),
    queryFn: ({ signal }) => fraudApi.readiness(signal),
    refetchInterval: 30_000,
    staleTime: 0,
    retry: false,
  });
}
