'use client';

import { ScanSearch } from 'lucide-react';
import { useMemo, useState } from 'react';
import { EmptyState } from '@/components/common/empty-state';
import { ErrorState } from '@/components/common/error-state';
import { PageHeader } from '@/components/common/page-header';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useSampleTransaction, useScoreTransaction } from '@/hooks';
import type { SampleKind } from '@/types';
import { EMPTY_FORM, FIELDS, fromTransaction, toPayload, type FieldName, type FormValues } from './fields';
import { ScoreResult } from './score-result';
import { TransactionForm } from './transaction-form';
import { ValidationErrors } from './validation-errors';

const FIELD_NAMES = new Set<string>(FIELDS.map((f) => f.name));

export function ScoreDashboard() {
  const [values, setValues] = useState<FormValues>(EMPTY_FORM);
  const score = useScoreTransaction();
  const sample = useSampleTransaction();

  const fieldErrors = useMemo(() => {
    const out: Partial<Record<FieldName, string>> = {};
    if (score.error?.kind !== 'validation') return out;
    for (const e of score.error.fieldErrors) {
      const name = String(e.loc[e.loc.length - 1]);
      if (FIELD_NAMES.has(name)) out[name as FieldName] = e.msg;
    }
    // Data-validation messages look like "amount: 1 values <= 0".
    for (const message of score.error.domainValidation?.errors ?? []) {
      const name = message.split(':')[0].trim();
      if (FIELD_NAMES.has(name)) out[name as FieldName] = message.slice(message.indexOf(':') + 1).trim();
    }
    return out;
  }, [score.error]);

  const loadSample = (kind: SampleKind) => {
    sample.mutate(kind, {
      onSuccess: (txn) => {
        setValues(fromTransaction(txn));
        score.reset();
      },
    });
  };

  return (
    <>
      <PageHeader
        title="Score a transaction"
        description="Send one payment to the model and see the fraud score, the decision band and the features that drove it."
      />
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Transaction</CardTitle>
            <CardDescription>
              Fields follow the assignment schema. Device and merchant history is looked up by the API, so a stored sample scores as it
              did in production.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TransactionForm
              values={values}
              onChange={(name, value) => setValues((v) => ({ ...v, [name]: value }))}
              onSubmit={() => score.mutate(toPayload(values))}
              onLoadSample={loadSample}
              onClear={() => {
                setValues(EMPTY_FORM);
                score.reset();
                sample.reset();
              }}
              fieldErrors={fieldErrors}
              isScoring={score.isPending}
              loadingSample={sample.isPending ? (sample.variables ?? null) : null}
              sampleError={sample.isError ? (sample.error.kind === 'network' ? 'API not reachable: start it with `make api`.' : sample.error.message) : null}
            />
          </CardContent>
        </Card>

        <div className="min-w-0 space-y-4">
          {score.isPending ? (
            <div role="status" className="space-y-4">
              <span className="sr-only">Scoring…</span>
              <Skeleton className="h-56 rounded-card" />
              <Skeleton className="h-64 rounded-card" />
            </div>
          ) : score.isError ? (
            score.error.kind === 'validation' ? (
              <ValidationErrors error={score.error} />
            ) : (
              <ErrorState error={score.error} title="Scoring failed" onRetry={() => score.mutate(toPayload(values))} />
            )
          ) : score.data ? (
            <ScoreResult result={score.data} />
          ) : (
            <EmptyState
              icon={<ScanSearch />}
              title="No score yet"
              description="Load a sample or fill in the form, then choose Score transaction. The decision, top reasons and explanation appear here."
              className="min-h-64"
            />
          )}
        </div>
      </div>
    </>
  );
}
