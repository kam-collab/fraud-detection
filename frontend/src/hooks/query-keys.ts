export const fraudKeys = {
  all: ['fraud'] as const,
  readiness: () => [...fraudKeys.all, 'readiness'] as const,
  overview: () => [...fraudKeys.all, 'overview'] as const,
  model: () => [...fraudKeys.all, 'model'] as const,
  monitoring: () => [...fraudKeys.all, 'monitoring'] as const,
  reviewQueue: () => [...fraudKeys.all, 'review-queue'] as const,
  reviewQueuePage: (limit: number, offset: number) => [...fraudKeys.reviewQueue(), { limit, offset }] as const,
  explain: (requestId: string, llm: boolean) => [...fraudKeys.all, 'explain', requestId, { llm }] as const,
};
