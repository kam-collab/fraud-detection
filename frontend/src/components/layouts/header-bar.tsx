import { ApiStatus } from '@/components/common/api-status';
import { ThemeToggle } from '@/components/common/theme-toggle';
import { Badge } from '@/components/ui/badge';
import { Brand } from './side-nav';

export function HeaderBar() {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-3 border-b bg-card/95 px-4 backdrop-blur sm:px-6">
      <div className="lg:hidden">
        <Brand />
      </div>
      <p className="hidden text-sm text-muted-foreground lg:block">Gradient-boosting fraud model · approve / manual review / block</p>
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="hidden sm:inline-flex">
          Synthetic data
        </Badge>
        <ApiStatus />
        <ThemeToggle />
      </div>
    </header>
  );
}
