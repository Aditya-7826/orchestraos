"""The UI's glassy "Tron" look: black glass panels with neon edges in one accent colour.

Every colour in the CSS comes from a few variables on :root, so switching theme only
swaps those variables. Streamlit's own dark base theme (.streamlit/config.toml) handles
the widgets this CSS doesn't touch.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    accent: str  # the neon colour: borders, glow, active states
    accent_2: str  # a darker partner for gradients


THEMES: dict[str, Theme] = {
    theme.name: theme
    for theme in [
        Theme("Red", "#ff2a3d", "#7a0010"),
        Theme("Cyan", "#18e6ff", "#005a73"),
        Theme("Green", "#2bff88", "#00602c"),
        Theme("Amber", "#ffb020", "#7a4a00"),
        Theme("Violet", "#b35cff", "#41107a"),
        Theme("Pink", "#ff3fb4", "#7a0a4c"),
        Theme("Ice", "#cfe9ff", "#3a5770"),
    ]
}
DEFAULT_THEME = "Red"
CUSTOM = "Custom"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def darken(color: str, factor: float = 0.45) -> str:
    r, g, b = (int(c * factor) for c in hex_to_rgb(color))
    return f"#{r:02x}{g:02x}{b:02x}"


def custom_theme(accent: str) -> Theme:
    return Theme(CUSTOM, accent, darken(accent))


def css(theme: Theme) -> str:
    r, g, b = hex_to_rgb(theme.accent)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

:root {{
  --accent: {theme.accent};
  --accent-2: {theme.accent_2};
  --accent-rgb: {r}, {g}, {b};
  --bg: #050507;
  --glass: rgba(14, 14, 20, 0.55);
  --glass-strong: rgba(18, 18, 26, 0.78);
  --edge: rgba(var(--accent-rgb), 0.35);
  --glow: 0 0 12px rgba(var(--accent-rgb), 0.35), inset 0 0 18px rgba(var(--accent-rgb), 0.06);
  --text: #e9e9f0;
  --muted: #9a9aab;
  --display: 'Orbitron', 'Segoe UI', system-ui, sans-serif;
}}

/* --- backdrop: black, a perspective grid on the floor, a glow on the horizon --- */
.stApp {{
  background:
    radial-gradient(ellipse 80% 45% at 50% 0%, rgba(var(--accent-rgb), 0.16), transparent 70%),
    linear-gradient(180deg, transparent 55%, rgba(var(--accent-rgb), 0.07) 100%),
    var(--bg);
  color: var(--text);
}}
.stApp::before {{
  content: "";
  position: fixed;
  inset: 45% -50% -10% -50%;
  background-image:
    linear-gradient(rgba(var(--accent-rgb), 0.22) 1px, transparent 1px),
    linear-gradient(90deg, rgba(var(--accent-rgb), 0.22) 1px, transparent 1px);
  background-size: 56px 56px;
  transform: perspective(420px) rotateX(62deg);
  transform-origin: top;
  mask-image: linear-gradient(180deg, transparent, #000 35%, #000 70%, transparent);
  pointer-events: none;
  z-index: 0;
}}
[data-testid="stHeader"] {{ background: var(--bg); }}
[data-testid="stMain"] {{ position: relative; z-index: 1; }}

/* --- type --- */
h1, h2, h3, [data-testid="stSidebar"] h1 {{
  font-family: var(--display) !important;
  letter-spacing: 0.06em;
  color: var(--text);
  text-shadow: 0 0 10px rgba(var(--accent-rgb), 0.55);
}}
a {{ color: var(--accent) !important; }}
code {{ color: var(--accent) !important; background: rgba(var(--accent-rgb), 0.08) !important; }}

/* --- glass panels --- */
[data-testid="stSidebar"] {{
  background: var(--glass-strong);
  backdrop-filter: blur(14px);
  border-right: 1px solid var(--edge);
  box-shadow: 4px 0 24px rgba(var(--accent-rgb), 0.12);
}}
[data-testid="stChatMessage"],
[data-testid="stExpander"] details,
[data-testid="stStatusWidget"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stAlertContainer"] {{
  background: var(--glass) !important;
  backdrop-filter: blur(12px);
  border: 1px solid var(--edge) !important;
  border-radius: 10px;
  box-shadow: var(--glow);
}}
[data-testid="stChatMessage"] {{ padding: 0.9rem 1rem; }}

/* --- inputs and buttons --- */
[data-testid="stChatInput"] > div,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {{
  background: var(--glass-strong) !important;
  border: 1px solid var(--edge) !important;
  border-radius: 10px !important;
}}
[data-testid="stChatInput"] > div:focus-within {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 16px rgba(var(--accent-rgb), 0.45);
}}
[data-testid="stBottom"] > div {{ background: transparent; }}
.stButton > button, .stDownloadButton > button {{
  font-family: var(--display);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.78rem;
  background: rgba(var(--accent-rgb), 0.06);
  color: var(--text);
  border: 1px solid var(--edge);
  border-radius: 8px;
  transition: box-shadow 0.2s, border-color 0.2s, background 0.2s;
}}
.stButton > button:hover {{
  border-color: var(--accent);
  color: #fff;
  box-shadow: 0 0 14px rgba(var(--accent-rgb), 0.55);
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border-color: var(--accent);
  color: #fff;
}}

/* --- the agent bar: stays pinned at the top while the conversation scrolls --- */
[data-testid="stElementContainer"]:has(.agent-bar) {{
  position: sticky;
  top: 3.75rem;  /* just below Streamlit's header */
  z-index: 50;
  padding: 0 0 0.4rem;
  background: linear-gradient(180deg, var(--bg) 70%, transparent);
}}
.agent-bar {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.6rem;
  margin: 0.8rem 0 0.6rem;
}}
.agent {{
  position: relative;
  min-width: 0;
  padding: 0.6rem 0.7rem;
  border-radius: 10px;
  background: linear-gradient(var(--glass), var(--glass)), var(--bg);
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: border-color 0.3s, box-shadow 0.3s, opacity 0.3s;
}}
.agent-head {{
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-family: var(--display);
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.agent-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: none;
  background: #44444f;
}}
.agent-name {{
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.agent-state {{
  position: absolute;
  top: -0.55rem;
  right: 0.6rem;
  padding: 0.05rem 0.4rem;
  border-radius: 4px;
  background: var(--bg);
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 0.52rem;
  letter-spacing: 0.1em;
  color: var(--muted);
}}
.agent.running .agent-state, .agent.waiting .agent-state {{
  color: var(--accent);
  border-color: var(--accent);
}}
.agent.waiting .agent-state {{ color: #ffb020; border-color: #ffb020; }}
.agent-line {{
  margin-top: 0.35rem;
  font-size: 0.74rem;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.agent.idle {{ opacity: 0.55; }}
.agent.queued {{ border-color: rgba(var(--accent-rgb), 0.25); }}
.agent.queued .agent-dot {{ background: rgba(var(--accent-rgb), 0.45); }}
.agent.running, .agent.waiting {{
  border-color: var(--accent);
  box-shadow: 0 0 18px rgba(var(--accent-rgb), 0.5), inset 0 0 14px rgba(var(--accent-rgb), 0.12);
}}
.agent.running .agent-line, .agent.waiting .agent-line {{ color: var(--text); }}
.agent.running .agent-dot {{ background: var(--accent); animation: pulse 1s ease-in-out infinite; }}
.agent.waiting .agent-dot {{ background: #ffb020; animation: pulse 1.6s ease-in-out infinite; }}
.agent.done .agent-dot {{ background: var(--accent); box-shadow: 0 0 6px var(--accent); }}
.agent.failed {{ border-color: rgba(255, 70, 70, 0.6); }}
.agent.failed .agent-dot {{ background: #ff4646; }}
.agent.running::after {{
  content: "";
  position: absolute;
  left: 10%;
  right: 10%;
  bottom: -1px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: scan 1.6s linear infinite;
}}
.status-line {{
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0.5rem 0.8rem;
  border-radius: 8px;
  /* opaque, so the conversation scrolling underneath doesn't show through */
  background: linear-gradient(rgba(var(--accent-rgb), 0.08), rgba(var(--accent-rgb), 0.08)),
    var(--bg);
  border-left: 2px solid var(--accent);
  font-size: 0.85rem;
  overflow: hidden;
}}
.status-line b {{
  font-family: var(--display);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  flex: none;
}}
.status-line span {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
@keyframes pulse {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(var(--accent-rgb), 0.7); }}
  50% {{ box-shadow: 0 0 0 6px rgba(var(--accent-rgb), 0); }}
}}
@keyframes scan {{
  0% {{ transform: scaleX(0.2); opacity: 0.4; }}
  50% {{ transform: scaleX(1); opacity: 1; }}
  100% {{ transform: scaleX(0.2); opacity: 0.4; }}
}}
@media (max-width: 640px) {{
  .agent-bar {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (prefers-reduced-motion: reduce) {{
  .agent.running .agent-dot, .agent.waiting .agent-dot, .agent.running::after {{ animation: none; }}
}}

/* --- theme swatches in the sidebar --- */
.swatches {{ display: flex; gap: 6px; margin: -0.3rem 0 0.6rem; }}
.swatch {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid rgba(255,255,255,.2); }}
.swatch.on {{ outline: 1px solid #fff; box-shadow: 0 0 8px currentColor; }}
</style>
"""


def swatches(current: Theme) -> str:
    """A row of little colour chips under the theme picker."""
    chips = "".join(
        f'<span class="swatch{" on" if t.accent == current.accent else ""}" '
        f'title="{t.name}" style="background:{t.accent};color:{t.accent}"></span>'
        for t in THEMES.values()
    )
    return f'<div class="swatches">{chips}</div>'
