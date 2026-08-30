/**
 * Handing a fetched file to the browser.
 *
 * The export endpoints return bytes rather than a URL, so there is nothing to
 * link to: the blob is turned into a temporary object URL, clicked, and
 * revoked. Doing it through a throwaway anchor keeps the page in place, which
 * a plain navigation would not.
 */
export function saveBlobAsFile(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
