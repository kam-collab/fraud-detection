import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';
import { Providers } from '@/lib/providers';
import './globals.css';

export const metadata: Metadata = {
  title: {
    default: 'Fraud Analyst Console',
    template: '%s · Fraud Analyst Console',
  },
  description: 'Analyst console for a payment fraud-detection model: scoring, manual review and drift monitoring. All data is synthetic.',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f9f9f7' },
    { media: '(prefers-color-scheme: dark)', color: '#0d0d0d' },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    // next-themes sets the theme class on <html> before hydration.
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-dvh">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
