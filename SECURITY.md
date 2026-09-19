# Security policy

This project evaluates agent behavior through a constrained tool interface. It is not a sandbox for
untrusted Python code. Local adapters run in the evaluator process and can access local files,
environment variables, and network resources. Use process or container isolation before running
untrusted adapters. See the [threat model](docs/threat-model.md) for the exact boundary.

The optional OpenAI adapter sends the public synthetic observation and tool schemas to the official
OpenAI API. Keep credentials in `OPENAI_API_KEY`, never in source files, prompts, reports, or issue
attachments. Provider calls have explicit request/output limits, a configured transport timeout,
and no automatic retries. Reports omit API keys and raw exception bodies; review any custom-agent
artifacts before sharing them.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/alinoori7723/locus-agent-evaluation/security/advisories/new)
for credential exposure, a validation bypass, or unintended disclosure. Include the affected commit,
a minimal reproduction, expected behavior, and impact. Do not include real credentials or private
provider responses. If private reporting is unavailable, open an issue asking for a private channel
without publishing exploit details or sensitive data.

Security fixes target the current default branch. Older snapshots remain useful for reproducing
historical reports but are not maintained release lines.
