import * as React from 'react';
import { cn } from '@/lib/utils';

/** Wide tables scroll inside their own container so the page body never does. */
function Table({ className, ...props }: React.ComponentProps<'table'>) {
  return (
    // `relative` keeps visually-hidden (absolutely positioned) cell content inside the scroll container.
    <div className="relative w-full overflow-x-auto">
      <table className={cn('w-full caption-bottom border-collapse text-sm', className)} {...props} />
    </div>
  );
}

function TableHeader({ className, ...props }: React.ComponentProps<'thead'>) {
  return <thead className={cn('[&_tr]:border-b', className)} {...props} />;
}

function TableBody({ className, ...props }: React.ComponentProps<'tbody'>) {
  return <tbody className={cn('[&_tr:last-child]:border-0', className)} {...props} />;
}

function TableRow({ className, ...props }: React.ComponentProps<'tr'>) {
  return <tr className={cn('border-b', className)} {...props} />;
}

function TableHead({ className, numeric, ...props }: React.ComponentProps<'th'> & { numeric?: boolean }) {
  return (
    <th
      scope="col"
      className={cn('whitespace-nowrap px-3 py-2 text-xs font-medium text-muted-foreground', numeric ? 'text-right' : 'text-left', className)}
      {...props}
    />
  );
}

function TableCell({ className, numeric, ...props }: React.ComponentProps<'td'> & { numeric?: boolean }) {
  return <td className={cn('px-3 py-2 align-middle', numeric && 'tabular text-right', className)} {...props} />;
}

function TableCaption({ className, ...props }: React.ComponentProps<'caption'>) {
  return <caption className={cn('px-3 pt-3 text-left text-xs text-muted-foreground', className)} {...props} />;
}

export { Table, TableHeader, TableBody, TableRow, TableHead, TableCell, TableCaption };
