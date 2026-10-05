import type { ReactNode } from 'react';
import { AppFooter } from './app-footer';
import { HeaderBar } from './header-bar';
import { MobileNav, SideNav } from './side-nav';

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-sm focus:text-primary-foreground"
      >
        Skip to content
      </a>
      <SideNav />
      <div className="flex min-w-0 flex-1 flex-col">
        <HeaderBar />
        <MobileNav />
        <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 space-y-6 px-4 py-6 sm:px-6">
          {children}
        </main>
        <AppFooter />
      </div>
    </div>
  );
}
