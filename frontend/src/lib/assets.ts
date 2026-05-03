const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function toAbsoluteUrl(pathOrUrl: string | null | undefined): string | null {
  if (!pathOrUrl) return null;

  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }

  if (pathOrUrl.startsWith("/")) {
    return `${API_BASE_URL}${pathOrUrl}`;
  }

  const normalized = pathOrUrl.replace(/\\/g, "/");
  const marker = "backend/storage/";
  if (normalized.includes(marker)) {
    const rel = normalized.split(marker)[1];
    return `${API_BASE_URL}/storage/${rel}`;
  }

  return `${API_BASE_URL}/${normalized.replace(/^\/+/, "")}`;
}
