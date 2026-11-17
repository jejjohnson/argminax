"""Draw argminax's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the whole idea in one picture: iterates descend across the level
sets of the cost J to its minimum x̂, and the filled ellipse there is the
uncertainty at the minimum, Ŝ = Ĥ⁻¹. vardax draws data assimilation over a
window and filterax sequential updates, so the minimisation picture is
argminax's. Standard library plus loguru; the tile and wordmark follow the
other repos of the stack (gaussx, filterax, functax).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

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

MINIMUM = (70.0, 70.0)  # x̂, the centre of the level sets
TILT = -30  # the level sets' tilt in degrees, so the cost couples the two axes
# (half-axes, stroke opacity) of each level set, outermost first
LEVEL_SETS = ((44.0, 26.0, 0.35), (28.0, 16.0, 0.5))
COVARIANCE = (13.0, 7.5)  # the half-axes of Ŝ's ellipse at x̂
ITERATES = ((24.0, 30.0), (46.0, 34.0), (52.0, 56.0), (64.0, 60.0), (67.0, 67.0))


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


def path(points: list[tuple[float, float]], *, width: float) -> str:
    commands = [
        f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(points)
    ]
    return (
        f'<path d="{" ".join(commands)}" fill="none" stroke="{WHITE}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def ellipse(
    half_axes: tuple[float, float],
    *,
    width: float,
    opacity: float,
    fill_opacity: float | None = None,
) -> str:
    centre_x, centre_y = MINIMUM
    radius_x, radius_y = half_axes
    fill = (
        f'fill="{WHITE}" fill-opacity="{fill_opacity}"'
        if fill_opacity is not None
        else 'fill="none"'
    )
    return (
        f'<ellipse cx="{centre_x}" cy="{centre_y}" rx="{radius_x}" ry="{radius_y}" '
        f'transform="rotate({TILT} {centre_x} {centre_y})" {fill} '
        f'stroke="{WHITE}" stroke-width="{width}" stroke-opacity="{opacity}"/>'
    )


def dot(x: float, y: float, radius: float) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{WHITE}"/>'


def glyph() -> str:
    """Level sets of J, the descent to x̂, and the covariance ellipse at x̂."""
    body = ""
    for radius_x, radius_y, opacity in LEVEL_SETS:
        body += ellipse((radius_x, radius_y), width=2.4, opacity=opacity)
    body += ellipse(COVARIANCE, width=2.8, opacity=0.95, fill_opacity=0.3)
    body += path([*ITERATES, MINIMUM], width=3.4)
    for x, y in ITERATES:
        body += dot(x, y, 3.6)
    return body + dot(*MINIMUM, 5.5)


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
