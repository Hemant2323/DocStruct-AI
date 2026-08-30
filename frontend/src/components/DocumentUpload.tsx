/**
 * Upload control: a drop zone, a file picker, and a tray showing the batch.
 *
 * Uploading returns as soon as the server accepts the file; extraction then runs
 * in the background. This component's job ends at "accepted" — the library takes
 * over from there and polls until each document finishes. The tray therefore
 * reports transfer only, and says so, rather than implying the document is ready
 * to use the moment the bytes land.
 *
 * Every file in a batch keeps its own row from queued through to accepted or
 * failed, so a partial failure names the file it belongs to instead of
 * collapsing into one message for the batch.
 */

import { useCallback, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { uploadDocument } from "@/api/documents";
import { Button } from "@/components/ui/Button";
import { StatusDot } from "@/components/ui/StatusDot";
import { formatBytes } from "@/lib/format";

/** Extensions offered in the picker. The server decides by content, not by name. */
const ACCEPTED = ".pdf,.docx,.txt,.csv,.xlsx";

type ItemState = "queued" | "uploading" | "accepted" | "failed";

interface UploadItem {
  /**
   * Identity within the batch. Not the filename: dropping two files with the
   * same name is legitimate, and keying on the name made both rows disappear
   * when the first one finished.
   */
  id: number;
  filename: string;
  sizeBytes: number;
  state: ItemState;
  message: string | null;
  requestId: string | null;
}

interface DocumentUploadProps {
  /** Called after each successful upload so the list can refresh. */
  onUploaded: () => void;
}

export function DocumentUpload({ onUploaded }: DocumentUploadProps): React.JSX.Element {
  const [isDragging, setIsDragging] = useState(false);
  const [items, setItems] = useState<UploadItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const nextItemId = useRef(1);

  const inFlight = items.some(
    (item) => item.state === "queued" || item.state === "uploading",
  );
  const accepted = items.filter((item) => item.state === "accepted").length;
  const failed = items.filter((item) => item.state === "failed").length;

  const updateItem = useCallback((id: number, patch: Partial<UploadItem>): void => {
    setItems((previous) =>
      previous.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  }, []);

  const upload = useCallback(
    async (files: FileList | File[]): Promise<void> => {
      const list = Array.from(files);
      if (list.length === 0) return;

      const batch: UploadItem[] = list.map((file) => ({
        id: nextItemId.current++,
        filename: file.name,
        sizeBytes: file.size,
        state: "queued",
        message: null,
        requestId: null,
      }));

      // The new batch replaces the last one's results — keeping them would leave
      // a growing log of files the user has already dealt with.
      setItems(batch);

      // Sequential on purpose: the server processes a bounded number at a time, and
      // firing everything at once would only queue them up while making failures
      // harder to attribute to a file.
      for (const [index, file] of list.entries()) {
        const item = batch[index];
        if (item === undefined) continue;

        updateItem(item.id, { state: "uploading" });
        try {
          await uploadDocument(file);
          updateItem(item.id, { state: "accepted" });
          onUploaded();
        } catch (error) {
          updateItem(item.id, {
            state: "failed",
            message:
              error instanceof ApiError
                ? error.message
                : "Upload failed. Please try again.",
            requestId: error instanceof ApiError ? error.requestId : null,
          });
        }
      }
    },
    [onUploaded, updateItem],
  );

  const openPicker = useCallback((): void => {
    inputRef.current?.click();
  }, []);

  return (
    <section className="border-line shrink-0 border-b p-3">
      <h2 className="text-ink-subtle mb-2 text-xs font-medium tracking-wide uppercase">
        Add documents
      </h2>

      {/* The zone is a pointer convenience only. The button inside it is the
          real control — focusable, labelled, and the only path a keyboard or
          screen-reader user needs. */}
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          // Dragging over a child fires `dragleave` on the parent. Without this
          // check the border flickered on and off across the zone's own text.
          if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          void upload(event.dataTransfer.files);
        }}
        onClick={(event) => {
          // Clicking the zone opens the picker, except on the button, which
          // opens it itself — forwarding both would open it twice.
          if (event.target instanceof Element && event.target.closest("button")) return;
          openPicker();
        }}
        className={`rounded-container cursor-pointer border border-dashed px-3 py-4 text-center transition-colors duration-100 ${
          isDragging ? "border-accent bg-accent-soft" : "border-line-strong"
        }`}
      >
        <p className="text-ink-muted text-xs">Drop files here</p>
        <Button size="sm" onClick={openPicker} className="mt-2">
          Choose files
        </Button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(event) => {
            if (event.target.files) void upload(event.target.files);
            // Reset so re-selecting the same file fires a change event again.
            event.target.value = "";
          }}
        />
        <p className="text-ink-subtle mt-2 text-xs">PDF, Word, Excel, CSV or text</p>
      </div>

      {items.length > 0 && (
        <div className="mt-3">
          <div className="mb-1.5 flex items-baseline justify-between gap-2">
            <p aria-live="polite" className="text-ink-muted text-xs">
              {inFlight
                ? `Uploading ${accepted + failed + 1} of ${items.length}…`
                : failed === 0
                  ? `${accepted} file${accepted === 1 ? "" : "s"} added — now processing`
                  : `${accepted} added, ${failed} failed`}
            </p>
            {!inFlight && (
              <button
                type="button"
                onClick={() => setItems([])}
                className="text-ink-subtle hover:text-ink focus-visible:outline-accent shrink-0 text-xs transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-1"
              >
                Clear
              </button>
            )}
          </div>

          {/* Capped: a fifty-file drop must not push the library off the screen. */}
          <ul className="max-h-44 space-y-1.5 overflow-y-auto">
            {items.map((item) => (
              <UploadRow key={item.id} item={item} />
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

/** One file's progress through the batch. */
function UploadRow({ item }: { item: UploadItem }): React.JSX.Element {
  const isFailed = item.state === "failed";

  return (
    <li className="text-xs">
      <div className="flex items-center gap-2">
        <StatusDot
          tone={isFailed ? "critical" : item.state === "accepted" ? "positive" : "muted"}
          pulse={item.state === "uploading"}
        />
        <span
          title={item.filename}
          className={`min-w-0 flex-1 truncate ${isFailed ? "text-critical" : "text-ink"}`}
        >
          {item.filename}
        </span>
        <span className="text-ink-subtle shrink-0 tabular-nums">
          {item.state === "uploading"
            ? "sending…"
            : item.state === "queued"
              ? "queued"
              : formatBytes(item.sizeBytes)}
        </span>
      </div>

      {isFailed && item.message !== null && (
        <p className="text-critical mt-0.5 pl-3.5">{item.message}</p>
      )}
      {isFailed && item.requestId !== null && (
        <p className="text-ink-subtle mt-0.5 pl-3.5 font-mono">
          Reference: {item.requestId}
        </p>
      )}
    </li>
  );
}
