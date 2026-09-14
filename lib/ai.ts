import { Doc, Frame, Group, Item, frameOfGroup } from "./tokens";
import { buildPrompt } from "./prompt";
import { isProject } from "./project";
import { Lang } from "./i18n";

/* Optional AI helpers. The browser talks to the model provider directly with the
 * author's own key; there is no server in between. Every action has a fixed
 * prompt and a fixed JSON answer shape, and the result is only applied after the
 * author has looked at it. Coordinates are never touched by the model. */

export type Provider = "claude" | "openai" | "metis" | "deepseek";

export type AiSettings = {
  provider: Provider;
  baseUrl: string;
  model: string;
  key: string;
};

export const PROVIDERS: {
  key: Provider;
  label: string;
  baseUrl: string;
  model: string;
  keysUrl?: string;
  /** this provider speaks the Anthropic /v1/messages dialect instead of OpenAI's */
  anthropic?: boolean;
  /** output token ceiling for requests to this provider; generous when the context window allows it */
  maxTokens?: number;
}[] = [
  { key: "openai", label: "OpenAI", baseUrl: "https://api.openai.com/v1", model: "gpt-5.6-luna", keysUrl: "https://platform.openai.com/api-keys" },
  { key: "claude", label: "Claude", baseUrl: "https://api.anthropic.com", model: "claude-sonnet-5", keysUrl: "https://console.anthropic.com/settings/keys", anthropic: true },
  { key: "metis", label: "Metis", baseUrl: "http://zsgw.sjdistributor.com:40000", model: "metis-coder-max", anthropic: true, maxTokens: 65536 },
  { key: "deepseek", label: "DeepSeek", baseUrl: "https://api.deepseek.com/v1", model: "deepseek-v4-flash", keysUrl: "https://platform.deepseek.com/api_keys" },
];

/** the output ceiling for a provider */
export const maxTokensFor = (k: Provider) => providerSpec(k).maxTokens ?? 8192;

export const providerSpec = (k: Provider) => PROVIDERS.find((p) => p.key === k) ?? PROVIDERS[0];

export const DEFAULT_AI: AiSettings = { provider: PROVIDERS[0].key, baseUrl: PROVIDERS[0].baseUrl, model: PROVIDERS[0].model, key: "" };

const STORE_KEY = "m3e:ai";

export function loadAiSettings(): AiSettings {
  const s = { ...DEFAULT_AI };
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) {
      const v = JSON.parse(raw) as Partial<AiSettings>;
      if (PROVIDERS.some((p) => p.key === v.provider)) s.provider = v.provider as Provider;
      if (typeof v.baseUrl === "string") s.baseUrl = v.baseUrl;
      if (typeof v.model === "string") s.model = v.model;
      if (typeof v.key === "string") s.key = v.key;
    }
  } catch {}
  return s;
}

/** Settings live in this browser only, like the document itself. */
export function saveAiSettings(s: AiSettings) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch {}
}

const isLocal = (u: string) => /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(u.trim());

/** hosts allowed over plain http; they only exist inside the company network */
const TRUSTED_HTTP_HOSTS = ["zsgw.sjdistributor.com"];

const isTrustedHttp = (u: string) => {
  try {
    return TRUSTED_HTTP_HOSTS.includes(new URL(u.trim()).hostname.toLowerCase());
  } catch {
    return false;
  }
};

/** a hosted endpoint needs a key; a server on this machine may run without one */
export const hasKey = (s: AiSettings) => s.key.trim().length > 0 || isLocal(s.baseUrl);

/** the key must not travel over plain http, except to this machine or a trusted internal gateway */
export const isSecureUrl = (u: string) => /^https:\/\//i.test(u.trim()) || isLocal(u) || isTrustedHttp(u);

const trimSlash = (u: string) => u.trim().replace(/\/+$/, "");

async function readError(res: Response): Promise<string> {
  let detail = "";
  try {
    const j = await res.json();
    detail = j?.error?.message ?? j?.message ?? JSON.stringify(j);
  } catch {
    try {
      detail = await res.text();
    } catch {}
  }
  return `${res.status} ${res.statusText}${detail ? `: ${detail.slice(0, 300)}` : ""}`;
}

/** the parsed body of an ok reply; when a gateway answers 2xx with something other than JSON, the raw payload goes into the error */
async function readBody(res: Response): Promise<any> {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`non-json ${res.status}: ${text.slice(0, 200).replace(/\s+/g, " ") || "(empty body)"}`);
  }
}

/** one round trip: a system prompt and a user message in, the model's text out */
export async function complete(s: AiSettings, system: string, user: string, signal?: AbortSignal, maxTokens = 4096): Promise<string> {
  const base = trimSlash(s.baseUrl);
  const model = s.model.trim();
  if (!model) throw new Error("model");
  if (!isSecureUrl(base)) throw new Error("insecure");
  const cap = maxTokensFor(s.provider);
  if (providerSpec(s.provider).anthropic) {
    const res = await fetch(`${base}/v1/messages`, {
      method: "POST",
      signal,
      headers: {
        "content-type": "application/json",
        "x-api-key": s.key.trim(),
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
      body: JSON.stringify({ model, max_tokens: Math.min(maxTokens, cap), system, messages: [{ role: "user", content: user }] }),
    });
    if (!res.ok) throw new Error(await readError(res));
    const j = await readBody(res);
    if (j.stop_reason === "refusal") throw new Error("refusal");
    if (j.stop_reason === "max_tokens") {
      /* the raw text rides along on the error, so a caller generating a big document can salvage what closed */
      const text = (j.content ?? []).filter((b: { type: string }) => b.type === "text").map((b: { text: string }) => b.text).join("");
      throw longError(text);
    }
    return (j.content ?? [])
      .filter((b: { type: string }) => b.type === "text")
      .map((b: { text: string }) => b.text)
      .join("");
  }
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (s.key.trim()) headers.authorization = `Bearer ${s.key.trim()}`;
  const res = await fetch(`${base}/chat/completions`, {
    method: "POST",
    signal,
    headers,
    body: JSON.stringify({
      model,
      /* OpenAI's newer models refuse `max_tokens` and default generously, so they get no budget;
         the other compatible endpoints cap around 8k */
      ...(s.provider === "openai" ? {} : { max_tokens: Math.min(maxTokens, cap) }),
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!res.ok) throw new Error(await readError(res));
  const j = await readBody(res);
  if (j.choices?.[0]?.finish_reason === "length") {
    /* the raw text rides along on the error, so a caller generating a big document can salvage what closed */
    const c0 = j.choices?.[0]?.message?.content;
    const text = typeof c0 === "string" ? c0 : Array.isArray(c0) ? c0.map((x: { text?: string }) => x.text ?? "").join("") : "";
    throw longError(text);
  }
  const c = j.choices?.[0]?.message?.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((x: { text?: string }) => x.text ?? "").join("");
  throw new Error("empty");
}

/** every balanced {...} span in a text, honoring strings; a `{` that never closes (a reasoning block, a cut reply) is skipped as a start */
function jsonSpans(text: string): string[] {
  const spans: string[] = [];
  for (let start = text.indexOf("{"); start >= 0; start = text.indexOf("{", start + 1)) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === "{") depth++;
      else if (ch === "}" && --depth === 0) {
        spans.push(text.slice(start, i + 1));
        break;
      }
    }
  }
  return spans;
}

/** a last-ditch parse for near-JSON: drops // and slash-star comments and dangling commas, the way models mis-write JSON */
function repairedParse(raw: string): unknown {
  let out = "";
  let inString = false;
  let escaped = false;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (inString) {
      out += ch;
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
      out += ch;
      continue;
    }
    if (ch === "/" && raw[i + 1] === "/") {
      while (i < raw.length && raw[i] !== "\n") i++;
      continue;
    }
    if (ch === "/" && raw[i + 1] === "*") {
      i += 2;
      while (i < raw.length && !(raw[i] === "*" && raw[i + 1] === "/")) i++;
      i++;
      continue;
    }
    if (ch === "}" || ch === "]") {
      let j = out.length - 1;
      while (j >= 0 && (out[j] === " " || out[j] === "\t" || out[j] === "\n" || out[j] === "\r")) j--;
      if (out[j] === ",") out = out.slice(0, j);
    }
    out += ch;
  }
  return JSON.parse(out);
}

/** a parse failure carries a slice of the raw reply, so the toast can show what the model actually wrote */
function jsonError(raw: string): Error {
  const e = new Error("json") as Error & { cause?: string };
  e.cause = raw.replace(/\s+/g, " ").trim().slice(0, 200);
  return e;
}

/** a cut-off reply keeps its full raw text, so a caller generating a big document can salvage what closed */
function longError(raw: string): Error {
  const e = new Error("long") as Error & { cause?: string };
  e.cause = raw;
  return e;
}

/** the JSON object in a reply; models talk around the answer (reasoning first, prose, fences,
 *  comments, dangling commas), so every balanced {...} is tried, leniently, before giving up.
 *  `valid` picks the span the caller can actually use when a reply holds more than one object. */
function parseJsonObject(text: string, valid?: (v: Record<string, unknown>) => boolean): Record<string, unknown> {
  const cleaned = text.replace(/```(?:json)?/gi, "");
  let fallback: Record<string, unknown> | undefined;
  for (const raw of jsonSpans(cleaned)) {
    let v: unknown;
    try {
      v = JSON.parse(raw);
    } catch {
      try {
        v = repairedParse(raw);
      } catch {
        continue;
      }
    }
    if (!v || typeof v !== "object" || Array.isArray(v)) continue;
    const o = v as Record<string, unknown>;
    if (!valid || valid(o)) return o;
    fallback ??= o;
  }
  if (fallback) return fallback;
  throw jsonError(text);
}

/* ---------- actions ---------- */

const LANG_NAME: Record<Lang, string> = { ja: "Japanese", en: "English", zh: "Simplified Chinese", ko: "Korean" };

const hasText = (v?: string | null) => !!v && v.trim().length > 0;

const itemsOnFrame = (doc: Doc, frame: Frame, widths: Record<string, number>): Item[] =>
  doc.groups.filter((g: Group) => frameOfGroup(g, doc.frames, widths)?.id === frame.id).flatMap((g) => g.items);

const describeItem = (it: Item) => {
  const bits = [`id=${it.id}`, `kind=${it.kind}`];
  if (hasText(it.label)) bits.push(`label=${JSON.stringify(it.label)}`);
  if (hasText(it.supporting)) bits.push(`supporting=${JSON.stringify(it.supporting)}`);
  if (it.icon) bits.push(`icon=${it.icon}`);
  if (it.tabs?.length) bits.push(`items=${JSON.stringify(it.tabs.map((t) => t.label || t.icon))}`);
  if (it.action) bits.push(`tap=${it.action.to}`);
  if (it.toggle) bits.push("toggle");
  if (hasText(it.note)) bits.push(`current_note=${JSON.stringify(it.note)}`);
  return bits.join(" ");
};

const context = (doc: Doc, widths: Record<string, number>, frame: Frame, lang: Lang) =>
  [
    "The author is sketching a Material 3 Expressive app. Below is the generated description of the whole design, then the screen to work on.",
    "",
    "=== Whole design ===",
    buildPrompt(doc, widths, undefined, lang),
    "",
    `=== Screen to work on: ${JSON.stringify(frame.name || "(unnamed)")} ===`,
  ].join("\n");

const SYSTEM = "You help an app designer finish a sketch. Answer with a single JSON object and nothing else: no prose, no markdown fence.";

/** picks the strings the model returned for exactly the parts asked about */
function pickStrings(v: unknown, parts: Item[], max: number): Record<string, string> {
  const map = (v ?? {}) as Record<string, unknown>;
  const out: Record<string, string> = {};
  for (const it of parts) {
    const s = map[it.id];
    if (typeof s === "string" && s.trim()) out[it.id] = s.trim().slice(0, max);
  }
  return out;
}

/** a short behavior note for one part; an existing note is refined rather than replaced */
export async function proposeBehavior(s: AiSettings, doc: Doc, widths: Record<string, number>, frame: Frame, lang: Lang, itemId: string, signal?: AbortSignal): Promise<string | undefined> {
  const parts = itemsOnFrame(doc, frame, widths).filter((it) => it.id === itemId);
  if (!parts.length) return undefined;
  const user = [
    context(doc, widths, frame, lang),
    "",
    "For the part listed below, write what happens when the user interacts with it: what it does, where it leads, what it shows. One sentence, concrete, in the voice of a product spec (no 'should', no hedging). Infer from the labels, icons and the other screens; do not invent screens that do not exist.",
    "A part with a current_note already has the author's own wording: keep its intent and facts, and improve it (clearer, more specific, consistent with the rest of the screen). Do not contradict it.",
    `Write in ${LANG_NAME[lang]}.`,
    "",
    "Part:",
    describeItem(parts[0]),
    "",
    'Answer as {"notes": {"<id>": "<sentence>"}} using exactly the id above.',
  ].join("\n");
  const j = parseJsonObject(await complete(s, SYSTEM, user, signal), (v) => v.notes != null && typeof v.notes === "object" && !Array.isArray(v.notes));
  return pickStrings(j.notes, parts, 300)[itemId];
}

/** a name (only when the screen has none) and a one-line purpose for the screen; an existing description is refined */
export async function proposeDescription(s: AiSettings, doc: Doc, widths: Record<string, number>, frame: Frame, lang: Lang, signal?: AbortSignal): Promise<{ name?: string; note: string }> {
  const user = [
    context(doc, widths, frame, lang),
    "",
    "Describe this screen's purpose in one or two sentences: who opens it, what they see and what they can do here. Also propose a short screen name (one to three words).",
    hasText(frame.note) ? `The author's current description is ${JSON.stringify(frame.note)}: keep its intent and facts, and improve it.` : "",
    `Write in ${LANG_NAME[lang]}.`,
    "",
    'Answer as {"name": "<name>", "description": "<sentences>"}.',
  ]
    .filter((l) => l !== "")
    .join("\n");
  const j = parseJsonObject(await complete(s, SYSTEM, user, signal), (v) => typeof v.description === "string");
  const note = typeof j.description === "string" ? j.description.trim().slice(0, 400) : "";
  if (!note) throw new Error("json");
  const name = typeof j.name === "string" ? j.name.trim().slice(0, 40) : "";
  return { name: !hasText(frame.name) && name ? name : undefined, note };
}

/** the value a rewritten field had before, so the rewrite can be undone; an empty field leaves nothing to go back to */
export const pushHistory = (history: string[] | undefined, replaced: string | undefined): string[] | undefined => {
  const cur = (replaced ?? "").trim();
  return cur ? [cur] : history?.length ? history : undefined;
};

/** the patch that swaps a field with what it said before the AI wrote it; pressing again swaps back */
export function popHistory<V extends string, H extends string>(current: string | undefined, history: string[] | undefined, valueKey: V, historyKey: H): Record<V, string> & Record<H, string[] | undefined> {
  const [prev] = history ?? [];
  const cur = (current ?? "").trim();
  return { [valueKey]: prev ?? "", [historyKey]: cur ? [cur] : undefined } as Record<V, string> & Record<H, string[] | undefined>;
}

/** A whole design from an idea, drafted by the author's own model. `guide` is the same
 *  agent guide a coding agent reads (public/agent.md), so both paths follow one spec.
 *  The answer is the document itself; a link would be pointless here. A reply cut off by
 *  the output budget is still kept when it closes into a readable document — the draft
 *  editor fixes the rest, so the author never starts from a blank screen. */
export async function draftDesign(s: AiSettings, guide: string, idea: string, lang: Lang, signal?: AbortSignal): Promise<Doc> {
  const system = [
    "You draft M3E Canvas designs. Follow the guide below exactly.",
    "Reply with the JSON document only: no share link, no prose, no markdown fence, no explanation.",
    "",
    guide,
  ].join("\n");
  const user = [
    `Sketch this app: ${idea.trim()}`,
    `Write every label, title and note in ${LANG_NAME[lang]}.`,
    "Three to five screens. Keep it simple.",
  ].join("\n");
  try {
    /* the draft may use the provider's whole output budget; a cut-off reply is salvaged below */
    const j = parseJsonObject(await complete(s, system, user, signal, maxTokensFor(s.provider)), isProject);
    if (!isProject(j)) throw new Error("json");
    return j;
  } catch (e) {
    /* a reply cut off by the output budget is still kept when it closes into a readable
       document — the draft editor fixes the rest, so the author never starts from a blank screen */
    if (e instanceof Error && e.message === "long" && typeof e.cause === "string" && e.cause) {
      try {
        const j = parseJsonObject(e.cause, isProject);
        if (isProject(j)) return j;
      } catch {}
    }
    throw e;
  }
}

