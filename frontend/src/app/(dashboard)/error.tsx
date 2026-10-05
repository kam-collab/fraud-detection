'use client';

import { ErrorState } from '@/components/common/error-state';

export default function DashboardError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <ErrorState error={error} onRetry={reset} title="This page failed to render" />;
}
