import type { Metadata } from 'next';
import { OverviewDashboard } from '@/components/dashboards/overview';

export const metadata: Metadata = { title: 'Overview' };

export default function OverviewPage() {
  return <OverviewDashboard />;
}
