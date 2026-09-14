import { afterEach, describe, expect, it, vi } from "vitest";

import { complete, draftDesign, proposeBehavior, type AiSettings } from "./ai";
import { BACK_TARGET, makeItem, type Doc, type Item } from "./tokens";
import { isProject } from "./project";

const settings: AiSettings = { provider: "metis", baseUrl: "http://zsgw.sjdistributor.com:40000", model: "metis-coder-max", key: "sk-test" };

const jsonReply = (body: unknown, status = 200) =>
  new Response(typeof body === "string" ? body : JSON.stringify(body), { status });

afterEach(() => vi.unstubAllGlobals());

/* One phone screen with a single button, so proposeBehavior has a part to ask about. */
function fixture(): Doc {
  const save: Item = { ...makeItem("button"), id: "save", label: "Save", action: { to: BACK_TARGET, transition: "fade" } };
  return {
    groups: [{ id: "g-row", x: 16, y: 400, axis: "x", items: [save] }],
    frames: [{ id: "f-home", name: "Home", x: 0, y: 0 }],
    paletteKey: "purple",
    frame: "phone",
    title: "Notes",
    brief: "",
  };
}

describe("complete", () => {
  it("sends the anthropic dialect to /v1/messages and joins the text blocks", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonReply({ content: [{ type: "text", text: "he" }, { type: "text", text: "llo" }] }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(complete(settings, "sys", "usr")).resolves.toBe("hello");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://zsgw.sjdistributor.com:40000/v1/messages");
    expect(init.headers["x-api-key"]).toBe("sk-test");
    expect(JSON.parse(init.body)).toMatchObject({ model: "metis-coder-max", system: "sys", messages: [{ role: "user", content: "usr" }] });
  });

  it("surfaces the raw payload when a 2xx reply is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>gateway error</html>", { status: 200 })));
    await expect(complete(settings, "s", "u")).rejects.toThrow(/non-json 200: <html>gateway error<\/html>/);
  });

  it("reads the provider's error message out of a failed reply", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401, statusText: "Unauthorized" })));
    await expect(complete(settings, "s", "u")).rejects.toThrow("401 Unauthorized: bad key");
  });

  it("caps the request at the provider's output ceiling", async () => {
    /* each dialect gets its own reply shape, chosen by the endpoint the request goes to */
    const fetchMock = vi.fn().mockImplementation((url: string) =>
      String(url).endsWith("/v1/messages") ? jsonReply({ content: [{ type: "text", text: "ok" }] }) : jsonReply({ choices: [{ message: { content: "ok" } }] }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await complete(settings, "s", "u", undefined, 20000);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).max_tokens).toBe(20000);
    await complete({ ...settings, provider: "deepseek", baseUrl: "https://api.deepseek.com/v1" }, "s", "u", undefined, 20000);
    expect(JSON.parse(fetchMock.mock.calls[1][1].body).max_tokens).toBe(8192);
  });

  it("salvages a truncated draft reply that still closes into a document", async () => {
    const doc = fixture();
    const full = JSON.stringify({ ...doc, groups: doc.groups, frames: [...doc.frames, { id: "f-second", name: "Second", x: 480, y: 0 }] });
    const cut = full.slice(0, full.lastIndexOf("}") + 1); /* drop trailing groups/frames items until braces stay balanced */
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonReply({ stop_reason: "max_tokens", content: [{ type: "text", text: cut }] })),
    );
    await expect(draftDesign(settings, "guide", "a notes app", "en")).resolves.toMatchObject({ title: doc.title });
  });

  it("still reports long when a truncated draft reply cannot be closed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonReply({ stop_reason: "max_tokens", content: [{ type: "text", text: '{"groups":[{"id":"g-1"' }] })),
    );
    await expect(draftDesign(settings, "guide", "a notes app", "en")).rejects.toThrow("long");
  });
});

describe("parseJsonObject via proposeBehavior", () => {
  const call = (reply: string) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonReply({ content: [{ type: "text", text: reply }] })));
    return proposeBehavior(settings, fixture(), {}, { id: "f-home", name: "Home", x: 0, y: 0 }, "zh", "save");
  };

  it("reads the answer out of a clean reply", async () => {
    await expect(call('{"notes":{"save":"点击后打开设置页。"}}')).resolves.toBe("点击后打开设置页。");
  });

  it("takes the first object when the model answers twice", async () => {
    await expect(call('{"notes":{"save":"第一句。"}}\n\n{"notes":{"save":"第二句。"}}')).resolves.toBe("第一句。");
  });

  it("ignores prose with braces around the answer", async () => {
    await expect(call('好的：{"notes":{"save":"一句话。"}}（花括号}示意）')).resolves.toBe("一句话。");
  });

  it("strips code fences", async () => {
    await expect(call('```json\n{"notes":{"save":"栅栏里的。"}}\n```')).resolves.toBe("栅栏里的。");
  });

  it("maps broken JSON to the friendly json error instead of leaking the parser message", async () => {
    await expect(call('{"notes":{"save": 没写完')).rejects.toThrow("json");
  });

  it("attaches a slice of the raw reply to the json error", async () => {
    try {
      await call("完全不是 JSON 的一段话");
      expect.unreachable("should have thrown");
    } catch (e) {
      const err = e as Error & { cause?: string };
      expect(err.message).toBe("json");
      expect(err.cause).toContain("完全不是 JSON");
    }
  });

  it("ignores strings when balancing braces", async () => {
    await expect(call('{"notes":{"save":"右括号}在文字里。"}} 之后的话')).resolves.toBe("右括号}在文字里。");
  });

  it("skips a reasoning preamble whose braces never close", async () => {
    await expect(call('我想想 { 先列要点：一、二\n{"notes":{"save":"推理后的答案。"}}')).resolves.toBe("推理后的答案。");
  });

  it("picks the usable object when a reasoning block holds well-formed JSON first", async () => {
    await expect(call('{"plan":"先写标题栏"}\n{"notes":{"save":"选中的答案。"}}')).resolves.toBe("选中的答案。");
  });

  it("repairs dangling commas", async () => {
    await expect(call('{"notes":{"save":"逗号后。",}}')).resolves.toBe("逗号后。");
  });

  it("repairs comments", async () => {
    await expect(call('{"notes": /* 首页的保存按钮 */ {"save":"注释后的答案。"}}')).resolves.toBe("注释后的答案。");
  });
});
