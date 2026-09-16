/**
 * An honest "nothing here" state. Used instead of a fabricated chart or
 * zero value whenever there is genuinely no data to show.
 */
export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="border border-dashed border-line px-4 py-6 text-sm text-ink-soft">
      <p className="font-medium text-ink">{title}</p>
      {detail && <p className="mt-1">{detail}</p>}
    </div>
  );
}
