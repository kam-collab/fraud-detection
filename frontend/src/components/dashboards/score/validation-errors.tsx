import { OctagonAlert } from 'lucide-react';
import type { ApiError } from '@/lib/api';
import { FIELDS } from './fields';

const LABELS = new Map<string, string>(FIELDS.map((f) => [f.name, f.label]));

/** Everything the API said in its HTTP 422 response, both schema and data validation. */
export function ValidationErrors({ error }: { error: ApiError }) {
  const fields = error.fieldErrors;
  const domain = error.domainValidation;

  return (
    <div role="alert" className="space-y-2 rounded-card border border-alert-fg/30 bg-alert-bg p-4 text-sm">
      <p className="flex items-center gap-2 font-semibold text-alert-fg">
        <OctagonAlert className="size-4 shrink-0" aria-hidden="true" />
        The API rejected this transaction (HTTP {error.status})
      </p>
      {fields.length > 0 ? (
        <ul className="list-disc space-y-0.5 pl-5">
          {fields.map((f, i) => {
            const name = String(f.loc[f.loc.length - 1] ?? 'body');
            return (
              <li key={`${name}-${i}`}>
                <span className="font-medium">{LABELS.get(name) ?? name}</span> <span className="font-mono text-xs text-subtle">({f.loc.join('.')})</span>:{' '}
                {f.msg}
              </li>
            );
          })}
        </ul>
      ) : null}
      {domain ? (
        <>
          {domain.errors.length > 0 ? (
            <ul className="list-disc space-y-0.5 pl-5">
              {domain.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          ) : null}
          {domain.warnings.length > 0 ? (
            <div>
              <p className="font-medium text-muted-foreground">Warnings</p>
              <ul className="list-disc space-y-0.5 pl-5 text-muted-foreground">
                {domain.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </>
      ) : null}
      {fields.length === 0 && !domain ? <p>{error.message}</p> : null}
    </div>
  );
}
