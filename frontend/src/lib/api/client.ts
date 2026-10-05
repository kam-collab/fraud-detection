/**
 * Minimal typed fetch wrapper for the FastAPI scoring service.
 * Every failure is normalised into an ApiError so the UI can tell
 * "API not reachable" apart from HTTP and validation errors.
 */
import type { DomainValidationDetail, FieldValidationError } from '@/types';

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

export type ApiErrorKind = 'network' | 'validation' | 'http';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number;
  /** Raw `detail` from the FastAPI error body, when there was one. */
  readonly detail: unknown;

  constructor(kind: ApiErrorKind, status: number, message: string, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }

  /** Per-field errors raised by request-schema validation. */
  get fieldErrors(): FieldValidationError[] {
    return Array.isArray(this.detail) ? (this.detail as FieldValidationError[]) : [];
  }

  /** Errors/warnings raised by the service's own data validation. */
  get domainValidation(): DomainValidationDetail | null {
    const d = this.detail;
    if (d && typeof d === 'object' && !Array.isArray(d) && 'errors' in d) {
      const { errors, warnings } = d as Partial<DomainValidationDetail>;
      return { errors: errors ?? [], warnings: warnings ?? [] };
    }
    return null;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

export function isNetworkError(error: unknown): boolean {
  return isApiError(error) && error.kind === 'network';
}

type QueryParams = Record<string, string | number | boolean | null | undefined>;

interface RequestOptions {
  method?: 'GET' | 'POST';
  params?: QueryParams;
  body?: unknown;
  signal?: AbortSignal;
}

function buildUrl(path: string, params?: QueryParams): string {
  const url = new URL(`${API_BASE_URL}${path}`);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== null && value !== undefined) url.searchParams.set(key, String(value));
  }
  return url.toString();
}

function describe(detail: unknown, status: number, statusText: string): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return `${detail.length} field${detail.length === 1 ? '' : 's'} failed validation`;
  if (detail && typeof detail === 'object' && 'errors' in detail) return 'The transaction failed data validation';
  return `Request failed (${status}${statusText ? ` ${statusText}` : ''})`;
}

export async function apiRequest<T>(path: string, { method = 'GET', params, body, signal }: RequestOptions = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(buildUrl(path, params), {
      method,
      signal,
      headers: body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError('network', 0, `API not reachable at ${API_BASE_URL}`);
  }

  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload ? (payload as { detail: unknown }).detail : payload;
    throw new ApiError(response.status === 422 ? 'validation' : 'http', response.status, describe(detail, response.status, response.statusText), detail);
  }
  return payload as T;
}
