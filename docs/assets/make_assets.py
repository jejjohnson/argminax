"""Draw argminax's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the whole idea in one picture: the cost J(x) is a valley, its
minimum x̂ is marked, and the Gaussian under it is the uncertainty there. The
sharper the valley (the larger the curvature Ĥ), the narrower the Gaussian
(Ŝ = Ĥ⁻¹). Standard library plus loguru; the tile and wordmark follow the other
repos of the stack (gaussx, filterax, functax).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

import math
from pathlib import Path

from loguru import logger


HERE = Path(__file__).resolve().parent

NAME = "argminax"
WORDMARK = ("argmina", "x")
LOGO_WIDTH = 560
# Slate: no other repo of the stack uses a neutral tile.
GRADIENT = ("#64748B", "#1E293B")
THEMES = {
    "light": {"ink": "#1F2937", "accent": "#475569"},
    "dark": {"ink": "#E5E7EB", "accent": "#94A3B8"},
}
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
WHITE = "#FFFFFF"

MINIMUM = (64.0, 62.0)  # x̂ on the valley
LEFT, RIGHT = 24.0, 104.0  # the glyph's horizontal extent
BASELINE = 100.0  # the zero line of the Gaussian
PEAK = 78.0  # the Gaussian's top, so its height is BASELINE - PEAK
SIGMA = 12.0  # the Gaussian's width in tile units


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{NAME}">\n'
        f"<title>{NAME}</title>\n{body}\n</svg>\n"
    )


def tile(gradient_id: str) -> str:
    start, stop = GRADIENT
    return (
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{stop}"/></linearGradient></defs>'
        '<rect x="4" y="4" width="120" height="120" rx="26" '
        f'fill="url(#{gradient_id})"/>'
    )


def path(
    points: list[tuple[float, float]],
    *,
    width: float,
    opacity: float = 1.0,
    close: bool = False,
    fill_opacity: float | None = None,
) -> str:
    commands = [
        f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(points)
    ]
    if close:
        commands.append("Z")
    fill = (
        f'fill="{WHITE}" fill-opacity="{fill_opacity}"'
        if fill_opacity is not None
        else 'fill="none"'
    )
    return (
        f'<path d="{" ".join(commands)}" {fill} stroke="{WHITE}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def glyph() -> str:
    """The cost valley, its minimum, and the Gaussian at the minimum."""
    centre, bottom = MINIMUM
    half = (RIGHT - LEFT) / 2
    steps = 40

    def valley(x: float) -> float:
        return bottom - 32 * ((x - centre) / half) ** 2

    def bell(x: float) -> float:
        return BASELINE - (BASELINE - PEAK) * math.exp(
            -0.5 * ((x - centre) / SIGMA) ** 2
        )

    xs = [LEFT + (RIGHT - LEFT) * i / steps for i in range(steps + 1)]
    curve = path([(x, valley(x)) for x in xs], width=5)
    outline = [(x, bell(x)) for x in xs]
    area = path(
        [(LEFT, BASELINE), *outline, (RIGHT, BASELINE)],
        width=0,
        close=True,
        fill_opacity=0.28,
    )
    top = path(outline, width=3, opacity=0.9)
    floor = path([(LEFT, BASELINE), (RIGHT, BASELINE)], width=3, opacity=0.8)
    drop = (
        f'<line x1="{centre}" y1="{bottom + 9}" x2="{centre}" y2="{PEAK - 4}" '
        f'stroke="{WHITE}" stroke-width="2" stroke-opacity="0.7" '
        'stroke-dasharray="2 4" stroke-linecap="round"/>'
    )
    marker = f'<circle cx="{centre}" cy="{bottom}" r="6.5" fill="{WHITE}"/>'
    return area + floor + top + curve + drop + marker


def icon() -> str:
    return svg(128, 128, tile(f"{NAME}-icon") + glyph())


def logo(theme: str) -> str:
    colours = THEMES[theme]
    head, tail = WORDMARK
    mark = f'<g transform="translate(6 6)">{tile(f"{NAME}-{theme}")}{glyph()}</g>'
    wordmark = (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{colours["ink"]}">{head}'
        f'<tspan fill="{colours["accent"]}">{tail}</tspan></text>'
    )
    return svg(LOGO_WIDTH, 140, mark + wordmark)


def main() -> None:
    outputs = {"icon.svg": icon()}
    for theme in THEMES:
        outputs[f"logo-{theme}.svg"] = logo(theme)
    for filename, content in outputs.items():
        (HERE / filename).write_text(content)
        logger.info("wrote {}", HERE / filename)


if __name__ == "__main__":
    main()
