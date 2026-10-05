/** One function per endpoint of the fraud-detection API. */
import { apiRequest } from './client';
import type {
  ExplainResult,
  Health,
  ModelReport,
  MonitoringReport,
  Overview,
  Readiness,
  ReviewDecisionResult,
  ReviewDecisionValue,
  ReviewQueue,
  SampleKind,
  ScoreResult,
  Transaction,
} from '@/types';

const V1 = '/api/v1';

export const fraudApi = {
  health: (signal?: AbortSignal) => apiRequest<Health>('/health', { signal }),
  readiness: (signal?: AbortSignal) => apiRequest<Readiness>('/readiness', { signal }),

  overview: (signal?: AbortSignal) => apiRequest<Overview>(`${V1}/overview`, { signal }),
  model: (signal?: AbortSignal) => apiRequest<ModelReport>(`${V1}/model`, { signal }),
  monitoring: (signal?: AbortSignal) => apiRequest<MonitoringReport>(`${V1}/monitoring`, { signal }),

  sampleTransaction: (kind: SampleKind) => apiRequest<Transaction>(`${V1}/transactions/sample`, { params: { kind } }),

  /** The body is deliberately loose: the form sends what the analyst typed and lets the API validate it. */
  score: (transaction: Record<string, unknown>) => apiRequest<ScoreResult>(`${V1}/score`, { method: 'POST', body: transaction }),

  reviewQueue: (limit: number, offset: number, signal?: AbortSignal) =>
    apiRequest<ReviewQueue>(`${V1}/review-queue`, { params: { limit, offset }, signal }),

  reviewDecision: (requestId: string, decision: ReviewDecisionValue) =>
    apiRequest<ReviewDecisionResult>(`${V1}/review-queue/${encodeURIComponent(requestId)}/decision`, {
      method: 'POST',
      body: { decision },
    }),

  explain: (requestId: string, llm: boolean, signal?: AbortSignal) =>
    apiRequest<ExplainResult>(`${V1}/explain/${encodeURIComponent(requestId)}`, { params: { llm }, signal }),
};
