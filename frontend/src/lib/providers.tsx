'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'next-themes';
import { useState, type ReactNode } from 'react';
import { isApiError } from '@/lib/api';

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Reports only change when the pipeline is re-run.
            staleTime: 5 * 60 * 1000,
            refetchOnWindowFocus: false,
            // Retry once when the API is unreachable; never retry a 4xx.
            retry: (failureCount, error) => failureCount < 1 && !(isApiError(error) && error.status >= 400 && error.status < 500),
          },
          mutations: { retry: false },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  );
}
