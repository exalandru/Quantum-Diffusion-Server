# QDS for Hermes

**QDS works natively with [Hermes](https://claude-code.nousresearch.com).** Not through a
bridge, not through a generic MCP adapter — two plugins that plug straight into
the parts of Hermes that were built for this.

Which means: you ask for an image in the chat, and your Mac makes it. No
account, no per-image cost, nothing leaving the machine.

There are two ways in. You can install one, or both — they don't overlap and
they don't interfere.

| | **Image provider** | **Playground plugin** |
|---|---|---|
| What it is | The engine behind Hermes' built-in image tool | A live studio beside the chat |
| You get | An image in the conversation | A window where the image forms as it renders |
| Best for | "Make me a picture of X" | Long renders, refining, upscaling, exploring |
| Folder | `image_gen/qds` | `qds_playground` |

---

## Option 1 — The image provider

**The simple one.** Hermes already knows how to generate images. This makes QDS
the thing that actually does it.

You ask, an image comes back in the conversation, and it's saved on your Mac.
That's the whole experience — no new commands to learn, no new surface to
manage. If you've ever used Hermes with a cloud image backend, this is the same
thing, except it's your hardware.

**Try it with:**

> Generate an image of a red fox asleep in a snowy forest, warm evening light.

---

## Option 2 — The playground plugin

**The one for when you want to watch.** Ask Hermes to open the playground and
it appears right beside the conversation, showing the image forming — blurry at
first, sharpening step by step, exactly like the QDS playground in your browser,
because it *is* the QDS playground.

The chat stays connected to that session. So you keep talking:

> Now make it wider.
> Try that again with a different seed.
> Upscale that one.

…and it all lands in the window you're already looking at.

Long renders stop being a problem. The generation belongs to the server, not to
whoever's watching it — close the window, go do something else, come back, and
it's exactly where it should be. A twenty-minute render is just a window you can
look away from.

**Try it with:**

> Open the QDS playground and start a cinematic wide shot of a lighthouse in a storm.

---

## Installing

Takes about a minute. You need [QDS](../README.md) installed and running, and
the Hermes desktop app.

### 1. Copy the plugins

Open the plugins folder: **Hermes → Settings → Plugins → Open plugins folder**.

Copy in whichever you want, keeping the folder names exactly as they are:

| Copy this folder | To |
|---|---|
| `hermes/image_gen/qds` | `plugins/image_gen/qds` |
| `hermes/qds_playground` | `plugins/qds_playground` |

The `image_gen` folder may not exist yet — create it.

### 2. Restart Hermes

Plugins are picked up when the app starts.

### 3. Turn them on

**Settings → Plugins → Agent plugins**, and flip the switch on:

- `qds` — the image provider
- `qds-playground` — the playground plugin

### 4. Point image generation at QDS

Only needed for the image provider. **Settings → Tools & Keys → Image
Generation**, choose **QDS (local)**, and pick a model if you'd like a specific
one.

No API key, no sign-in. It's your machine.

### That's it

Ask for an image. If you installed the playground plugin, ask Hermes to open the
playground and watch it work.

---

## If something's off

| What you see | What it means |
|---|---|
| Hermes says QDS isn't reachable | The server isn't running. Start QDS from the menu bar. |
| The plugins don't show up | Check the folder names, then restart Hermes — the folder is only read at startup. |
| The model list looks empty | QDS was asleep when Hermes last looked. Restart the server, then reopen the settings panel. |
| The first image takes ages | The model weights are loading. It's a one-time cost per model — the next image is fast. Small models like `anima-turbo` load in seconds. |
| A generation won't start | The playground queue may be paused. The pane has a resume button. |
| A session asks for a password | You locked it in the QDS playground. Unlock it there. |

**Running QDS somewhere unusual?** Set `QDS_BASE_URL` (defaults to
`http://127.0.0.1:8765`), and `QDS_API_KEY` if you gave your server a key. A
normal local install needs neither.

Neither plugin can install models, change your server config, or restart
anything — that all stays in QDS, where it belongs.
