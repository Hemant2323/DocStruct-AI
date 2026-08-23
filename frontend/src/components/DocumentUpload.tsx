/**
 * Upload control: a drop zone plus a file picker.
 *
 * Uploading returns as soon as the server accepts the file; extraction then runs in the
 * background. This component's job ends at "accepted" — the list takes over from there
 * and polls until each document finishes.
 */

import { useCallback, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { uploadDocument } from "@/api/documents";

/** Extensions offered in the picker. The server decides by content, not by name. */
const ACCEPTED = ".pdf,.docx,.txt,.csv,.xlsx";

interface DocumentUploadProps {
  /** Called after each successful upload so the list can refresh. */
  onUploaded: () => void;
}

interface FailedUpload {
  filename: string;
  message: string;
  requestId: string | null;
}

export function DocumentUpload({
  onUploaded,
}: DocumentUploadProps): React.JSX.Element {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState<string[]>([]);
  const [failures, setFailures] = useState<FailedUpload[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (files: FileList | File[]): Promise<void> => {
      const list = Array.from(files);
      if (list.length === 0) return;

      setFailures([]);
      setUploading(list.map((file) => file.name));

      // Sequential on purpose: the server processes a bounded number at a time, and
      // firing everything at once would only queue them up while making failures
      // harder to attribute to a file.
      for (const file of list) {
        try {
          await uploadDocument(file);
          onUploaded();
        } catch (error) {
          setFailures((previous) => [
            ...previous,
            {
              filename: file.name,
              message:
                error instanceof ApiError
                  ? error.message
                  : "Upload failed. Please try again.",
              requestId: error instanceof ApiError ? error.requestId : null,
            },
          ]);
        } finally {
          setUploading((previous) => previous.filter((name) => name !== file.name));
        }
      }
    },
    [onUploaded],
  );

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-slate-800">
      <h2 className="mb-4 text-lg font-semibold text-slate-900 dark:text-slate-100">
        Upload documents
      </h2>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          void upload(event.dataTransfer.files);
        }}
        className={`rounded-lg border-2 border-dashed p-8 text-center transition ${
          isDragging
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950"
            : "border-slate-300 dark:border-slate-600"
        }`}
      >
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Drag files here, or
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="mt-3 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
        >
          Choose files
        </button>
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
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-500">
          PDF, Word, Excel, CSV or plain text
        </p>
      </div>

      {uploading.length > 0 && (
        <ul className="mt-4 space-y-1">
          {uploading.map((name) => (
            <li
              key={name}
              className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400"
            >
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-500"
              />
              Uploading {name}…
            </li>
          ))}
        </ul>
      )}

      {failures.length > 0 && (
        <ul className="mt-4 space-y-2">
          {failures.map((failure) => (
            <li
              key={failure.filename}
              className="rounded-md border border-red-200 bg-red-50 p-3 text-sm dark:border-red-900 dark:bg-red-950"
            >
              <p className="font-medium text-red-800 dark:text-red-200">
                {failure.filename}
              </p>
              <p className="mt-1 text-red-700 dark:text-red-300">
                {failure.message}
              </p>
              {failure.requestId !== null && (
                <p className="mt-1 font-mono text-xs text-red-600 dark:text-red-400">
                  {failure.requestId}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
