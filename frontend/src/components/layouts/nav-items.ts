import { Activity, LayoutDashboard, ListChecks, ScanSearch, type LucideIcon } from 'lucide-react';

export interface NavItem {
  href: string;
  label: string;
  description: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavItem[] = [
  { href: '/', label: 'Overview', description: 'Model quality and data', icon: LayoutDashboard },
  { href: '/score', label: 'Score a transaction', description: 'Score, band and reasons', icon: ScanSearch },
  { href: '/review', label: 'Review queue', description: 'Approve or decline', icon: ListChecks },
  { href: '/monitoring', label: 'Drift monitoring', description: 'July to September', icon: Activity },
];
