import { Plugin } from "@opencode/plugin"
import { execFile } from "node:child_process"
import { promisify } from "node:util"

const run = promisify(execFile)
const ROOT = new URL("../../", import.meta.url).pathname
const DEFAULT_MODEL = "/tmp/qwen2-0.5b-q4km.gguf"

type BenchArgs = {
  model?: string
  prompt?: string
  n?: number
  device?: string
  graph?: boolean
  threads?: number
}

async function bench(args: BenchArgs) {
  const cmd = [
    `${ROOT}target/release/examples/bench`,
    args.model || DEFAULT_MODEL,
    args.prompt || "Once upon a time",
    String(args.n ?? 40),
    "--device", args.device || "cuda",
    "--threads", String(args.threads ?? 0),
  ]
  const env = { ...process.env }
  if (args.graph) env.CORTEX_CUDA_GRAPH = "1"
  try {
    const { stdout, stderr } = await run(cmd[0], cmd.slice(1), {
      cwd: ROOT,
      env,
      timeout: 600_000,
      maxBuffer: 8 * 1024 * 1024,
    })
    return { content: stdout + (stderr ? `\n[stderr]\n${stderr}` : "") }
  } catch (e: any) {
    return { content: `bench failed: ${e.stdout || ""}\n${e.stderr || e.message}` }
  }
}

export default Plugin.define({
  id: "cortex.helpers",
  async setup(ctx) {
    await ctx.tool.transform((editor) => {
      editor.namespace({
        name: "cortex",
        description: "Cortex build/benchmark/parity helpers",
      })
      editor.add({
        name: "bench",
        description: "Benchmark decode throughput. graph=true enables CORTEX_CUDA_GRAPH.",
        input: {
          type: "object",
          properties: {
            model: { type: "string" },
            prompt: { type: "string" },
            n: { type: "integer" },
            device: { type: "string" },
            threads: { type: "integer" },
            graph: { type: "boolean" },
          },
          additionalProperties: false,
        },
        options: { namespace: "cortex", codemode: true },
        execute: async (input: any) => bench(input as BenchArgs),
      })
      editor.add({
        name: "test",
        description: "Run cargo test for cortex-llm",
        input: { type: "object", properties: {}, additionalProperties: false },
        options: { namespace: "cortex", codemode: true },
        execute: async () => {
          try {
            const { stdout, stderr } = await run("cargo", ["test"], { cwd: ROOT, timeout: 600_000 })
            return { content: stdout + stderr }
          } catch (e: any) {
            return { content: `test failed: ${e.stdout || ""}\n${e.stderr || e.message}` }
          }
        },
      })
      editor.add({
        name: "status",
        description: "Show git status and HEAD of cortex-llm",
        input: { type: "object", properties: {}, additionalProperties: false },
        options: { namespace: "cortex", codemode: true },
        execute: async () => {
          const a = await run("git", ["status", "--short"], { cwd: ROOT })
          const b = await run("git", ["log", "-1", "--oneline"], { cwd: ROOT })
          return { content: `${a.stdout}\n${b.stdout}` }
        },
      })
    })
  },
})
