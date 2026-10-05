import type { Metadata } from 'next';
import { MonitoringDashboard } from '@/components/dashboards/monitoring';

export const metadata: Metadata = { title: 'Drift monitoring' };

export default function MonitoringPage() {
  return <MonitoringDashboard />;
}
