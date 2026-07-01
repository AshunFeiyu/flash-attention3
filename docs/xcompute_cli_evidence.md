# XCompute CLI Evidence

Use `xcu` as the default SQTT/Wavefronts evidence path for every perf
candidate.  The GUI is allowed only as a fallback or manual cross-check.

Tool guide:

```text
/Volumes/172.20.68.76/共享/工具/sqtt-cli-agent-guide.md
```

Sidecar package candidates:

```text
${SHAOBO_RUN_ROOT}/tools/XCompute-Light-4.6.3-Linux-sqtt-cli.deb
/Volumes/172.20.68.76/共享/工具/XCompute-Light-4.6.3-Linux-sqtt-cli.deb
/共享/工具/XCompute-Light-4.6.3-Linux-sqtt-cli.deb
```

## Workflow

1. Run `scripts/xcu_preflight.sh`.
2. Capture PMD perf with SQTT enabled.
3. Run a first pass:

   ```bash
   scripts/analyze_sqtt_perf.sh --perf case.perf --dispatch 1
   ```

4. Read `wavefronts_bubbles.txt`, choose a `Start:End` window and a concrete
   `xcd/se/cu/simd/wave` location.
5. Export full pipeline and SIMD CSV:

   ```bash
   scripts/analyze_sqtt_perf.sh --perf case.perf --dispatch 1 \
     --time-range 4148:10828 \
     --location xcd=0,se=0,cu=6,simd=1,wave=1
   ```

## Required Artifacts

Each promoted perf run must keep these files outside the git repo:

- `detail.txt`
- `wavefronts_bubbles.txt`
- `pipeline/` CSV directory when a time window is selected
- `simd/` CSV directory when a time window is selected
- `manifest.md`

## Interpretation Checklist

- Does `detail` show the intended dispatch and kernel?
- What are the top opcodes and issue-bubble pairs?
- Are MMAC islands continuous or broken by `s_waitcnt`, LDS, VALU, barrier, or
  instruction-fetch/no-v bubbles?
- Do both consumer groups show useful overlap, or do they still step together?
- Is the thin producer problem visible in wavefront instruction counts?
- Does SIMD mix explain MMAC active share changing up or down?
