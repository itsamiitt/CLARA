---
description: Show CLARA's live memory counter in the Claude Code status bar
argument-hint: [on|off]
---

Turn CLARA's status-bar memory counter on or off.

Read `$ARGUMENTS`: treat "off", "remove", "disable", or "hide" as turning it
off; anything else (including no argument) means turning it on.

1. Call `statusline_status` first so you can tell the user what is already
   configured.
2. To turn it on, call `statusline_install` with `enable: true`. To turn it
   off, call it with `enable: false`.
3. Interpret the result rather than dumping the JSON:
   - `installed` — tell them it is set up and that the counter appears in the
     status bar **after they start a new Claude Code session**.
   - `removed` / `absent` — confirm it is gone, or was never configured.
   - `blocked` — they already use a different status line. Show the existing
     command and ask whether to replace it; only re-run with `force: true` if
     they say yes. Never force without asking.
   - `error` — relay the message plainly (usually a malformed
     `~/.claude/settings.json` they need to fix).

Their existing settings are preserved and backed up automatically, so there is
no need to warn them about losing configuration.
