'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';
import { NAV_ITEMS } from './nav-items';

function isActive(pathname: string, href: string) {
  return href === '/' ? pathname === '/' : pathname === href || pathname.startsWith(`${href}/`);
}

export function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2.5 rounded-md">
      <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
        <ShieldCheck className="size-4.5" aria-hidden="true" />
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-semibold">Fraud Analyst Console</span>
        <span className="block text-xs text-muted-foreground">Payments risk</span>
      </span>
    </Link>
  );
}

/** Vertical navigation shown from the `lg` breakpoint up. */
export function SideNav() {
  const pathname = usePathname();
  return (
    <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r bg-card lg:flex">
      <div className="flex h-14 items-center border-b px-4">
        <Brand />
      </div>
      <nav aria-label="Main" className="flex-1 space-y-1 overflow-y-auto p-3">
        {NAV_ITEMS.map(({ href, label, description, icon: Icon }) => {
          const active = isActive(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? 'page' : undefined}
              className={cn(
                'flex items-start gap-3 rounded-lg border-l-2 px-3 py-2.5 transition-colors',
                active ? 'border-primary bg-accent text-foreground' : 'border-transparent text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
            >
              <Icon className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
              <span className="leading-tight">
                <span className={cn('block text-sm', active ? 'font-semibold' : 'font-medium')}>{label}</span>
                <span className="block text-xs text-subtle">{description}</span>
              </span>
            </Link>
          );
        })}
      </nav>
      <p className="border-t p-4 text-xs leading-relaxed text-subtle">Synthetic data. No real customers, merchants or payments.</p>
    </aside>
  );
}

/** Horizontal navigation shown below the `lg` breakpoint. */
export function MobileNav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex gap-1 overflow-x-auto border-b bg-card px-3 py-2 lg:hidden">
      {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
        const active = isActive(pathname, href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors',
              active ? 'bg-accent font-semibold text-foreground ring-1 ring-primary/40' : 'font-medium text-muted-foreground hover:bg-muted'
            )}
          >
            <Icon className="size-4" aria-hidden="true" />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
