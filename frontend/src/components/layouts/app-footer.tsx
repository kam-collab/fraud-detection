import { API_BASE_URL } from '@/lib/api';

export function AppFooter() {
  return (
    <footer className="border-t px-4 py-4 text-xs leading-relaxed text-subtle sm:px-6">
      <p>
        <strong className="font-semibold text-muted-foreground">All data is synthetic.</strong> Transactions, devices, merchants and
        fraud labels are produced by a seeded generator for a take-home assignment; nothing here describes real customers or payments.
      </p>
      <p className="mt-1">
        API: <span className="font-mono">{API_BASE_URL}</span>
      </p>
    </footer>
  );
}
