'use client';

import { Loader2, ScanSearch } from 'lucide-react';
import { useId } from 'react';
import { Button } from '@/components/ui/button';
import { Input, Select } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { SampleKind } from '@/types';
import { FIELDS, type FieldName, type FormValues } from './fields';

const SAMPLES: { kind: SampleKind; label: string }[] = [
  { kind: 'random', label: 'Random' },
  { kind: 'fraud', label: 'Fraud' },
  { kind: 'genuine', label: 'Genuine' },
  { kind: 'review', label: 'Review band' },
  { kind: 'blocked', label: 'Blocked' },
];

interface TransactionFormProps {
  values: FormValues;
  onChange: (name: FieldName, value: string) => void;
  onSubmit: () => void;
  onLoadSample: (kind: SampleKind) => void;
  onClear: () => void;
  /** Messages keyed by field, from the API's 422 response. */
  fieldErrors: Partial<Record<FieldName, string>>;
  isScoring: boolean;
  loadingSample: SampleKind | null;
  sampleError: string | null;
}

export function TransactionForm({ values, onChange, onSubmit, onLoadSample, onClear, fieldErrors, isScoring, loadingSample, sampleError }: TransactionFormProps) {
  const id = useId();

  return (
    <form
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="space-y-5"
    >
      <fieldset className="space-y-2">
        <legend className="text-xs font-medium text-muted-foreground">Load a stored September transaction</legend>
        <div className="flex flex-wrap gap-2">
          {SAMPLES.map(({ kind, label }) => (
            <Button key={kind} variant="outline" size="sm" onClick={() => onLoadSample(kind)} disabled={loadingSample !== null}>
              {loadingSample === kind ? <Loader2 className="animate-spin" aria-hidden="true" /> : null}
              {label}
            </Button>
          ))}
        </div>
        <p role="status" className="text-xs text-alert-fg">
          {sampleError}
        </p>
      </fieldset>

      <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2">
        {FIELDS.map((field) => {
          const fieldId = `${id}-${field.name}`;
          const error = fieldErrors[field.name];
          const describedBy = [error ? `${fieldId}-error` : null, field.hint ? `${fieldId}-hint` : null].filter(Boolean).join(' ') || undefined;
          const common = {
            id: fieldId,
            name: field.name,
            value: values[field.name],
            'aria-invalid': error ? true : undefined,
            'aria-required': field.required || undefined,
            'aria-describedby': describedBy,
          };
          return (
            <div key={field.name} className="space-y-1">
              <Label htmlFor={fieldId}>
                {field.label}
                {field.required ? (
                  <span className="text-alert-fg" aria-hidden="true">
                    {' '}
                    *
                  </span>
                ) : null}
              </Label>
              {field.options ? (
                <Select {...common} onChange={(e) => onChange(field.name, e.target.value)}>
                  {/* Keep a value that came from a sample or was cleared, even if it is not a known option. */}
                  {field.options.some((o) => o.value === values[field.name]) ? null : (
                    <option value={values[field.name]}>{values[field.name] || 'Select…'}</option>
                  )}
                  {field.options.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              ) : (
                <Input
                  {...common}
                  type="text"
                  inputMode={field.type === 'number' ? 'decimal' : undefined}
                  autoComplete="off"
                  spellCheck={false}
                  placeholder={field.placeholder}
                  onChange={(e) => onChange(field.name, e.target.value)}
                />
              )}
              {error ? (
                <p id={`${fieldId}-error`} className="text-xs font-medium text-alert-fg">
                  {error}
                </p>
              ) : field.hint ? (
                <p id={`${fieldId}-hint`} className="text-xs text-subtle">
                  {field.hint}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button type="submit" disabled={isScoring}>
          {isScoring ? <Loader2 className="animate-spin" aria-hidden="true" /> : <ScanSearch aria-hidden="true" />}
          Score transaction
        </Button>
        <Button variant="ghost" onClick={onClear} disabled={isScoring}>
          Clear
        </Button>
        <p className="text-xs text-subtle">
          <span aria-hidden="true">* </span>Required by the API.
        </p>
      </div>
    </form>
  );
}
