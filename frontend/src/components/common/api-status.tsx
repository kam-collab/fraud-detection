'use client';

import { CheckCircle2, Loader2, ServerCrash } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { useReadiness } from '@/hooks';

/** Header pill: is the scoring API up and which model version is loaded. */
export function ApiStatus() {
  const { data, isPending, isError } = useReadiness();

  if (isPending) {
    return (
      <Badge variant="neutral">
        <Loader2 className="animate-spin" aria-hidden="true" />
        Checking API
      </Badge>
    );
  }
  if (isError || !data) {
    return (
      <Badge variant="alert" title="Start the API with `make api`">
        <ServerCrash aria-hidden="true" />
        API offline
      </Badge>
    );
  }
  return (
    <Badge variant="ok" title={`${data.history_rows.toLocaleString('en-IN')} transactions loaded`}>
      <CheckCircle2 aria-hidden="true" />
      <span>
        API {data.status}
        <span className="hidden sm:inline"> · model v{data.model_version}</span>
      </span>
    </Badge>
  );
}
