import * as React from 'react';
import { cn } from '@/lib/utils';

const fieldClass =
  'h-9 w-full rounded-md border border-input bg-card px-3 text-sm text-foreground placeholder:text-subtle disabled:opacity-50 aria-[invalid=true]:border-alert-fg';

function Input({ className, ...props }: React.ComponentProps<'input'>) {
  return <input className={cn(fieldClass, className)} {...props} />;
}

function Select({ className, ...props }: React.ComponentProps<'select'>) {
  return <select className={cn(fieldClass, 'pr-8', className)} {...props} />;
}

export { Input, Select };
