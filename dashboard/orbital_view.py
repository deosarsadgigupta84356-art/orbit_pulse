
"""
ORBIT PULSE - 3D Orbital Intelligence
=====================================
NASA Blue Marble textured Earth with an illustrative conjunction scene.

IMPORTANT:
The Earth texture is a real NASA image. Object positions and orbital
trajectories are deterministic illustrations, NOT real ephemerides.
The current synthetic CDM fixture does not contain orbital state vectors.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image


BACKGROUND = "#050B18"
PANEL = "#101B31"
CYAN = "#00E5FF"
BLUE = "#38BDF8"
VIOLET = "#C084FC"
MUTED = "#94A3B8"
WHITE = "#F8FAFC"

EARTH_TEXTURE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "earth"
    / "earth_texture.jpg"
)


def _stable_parameters(object_id):
    """Create repeatable illustrative orbit parameters."""
    digest = hashlib.sha256(
        str(object_id).encode("utf-8")
    ).digest()

    return {
        "inclination": np.radians(20 + digest[0] / 255 * 65),
        "raan": np.radians(digest[1] / 255 * 360),
        "phase": np.radians(digest[2] / 255 * 360),
        "radius": 1.55 + digest[3] / 255 * 0.30,
    }


def _orbit_coordinates(parameters, samples=240):
    """Generate a schematic circular orbit."""
    angle = np.linspace(0, 2 * np.pi, samples)
    radius = parameters["radius"]

    coordinates = np.vstack((
        radius * np.cos(angle),
        radius * np.sin(angle),
        np.zeros_like(angle),
    ))

    inclination = parameters["inclination"]
    raan = parameters["raan"]

    rotate_x = np.array([
        [1, 0, 0],
        [0, np.cos(inclination), -np.sin(inclination)],
        [0, np.sin(inclination), np.cos(inclination)],
    ])

    rotate_z = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1],
    ])

    result = rotate_z @ rotate_x @ coordinates
    return result[0], result[1], result[2]


def _object_position(parameters):
    """Return one illustrative position on the generated orbit."""
    angle = parameters["phase"]
    radius = parameters["radius"]

    position = np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        0.0,
    ])

    inclination = parameters["inclination"]
    raan = parameters["raan"]

    rotate_x = np.array([
        [1, 0, 0],
        [0, np.cos(inclination), -np.sin(inclination)],
        [0, np.sin(inclination), np.cos(inclination)],
    ])

    rotate_z = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1],
    ])

    return rotate_z @ rotate_x @ position


def _object_category(object_id):
    """Classify objects using their CDM identifier prefixes."""
    normalized = str(object_id).strip().upper()

    if normalized.startswith("SAT"):
        return "SATELLITE"

    if normalized.startswith("DEB"):
        return "DEBRIS"

    return "OTHER"


def _is_visible(category, show_satellites, show_debris):
    if category == "SATELLITE":
        return show_satellites

    if category == "DEBRIS":
        return show_debris

    return show_satellites and show_debris


@lru_cache(maxsize=2)
def _earth_mesh_data(texture_path):
    """Load NASA's equirectangular texture and map it onto a 3D sphere."""
    image_path = Path(texture_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Earth texture not found: {image_path}"
        )

    # A 180 x 90 mesh keeps rendering responsive.
    n_lon = 180
    n_lat = 90

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((720, 360), Image.Resampling.LANCZOS)
        texture = np.asarray(image, dtype=np.uint8)

    longitude = np.linspace(
        -np.pi, np.pi, n_lon, endpoint=False
    )
    latitude = np.linspace(
        np.pi / 2, -np.pi / 2, n_lat
    )

    lon_grid, lat_grid = np.meshgrid(longitude, latitude)

    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    image_height, image_width, _ = texture.shape

    pixel_x = (
        (lon_grid + np.pi) / (2 * np.pi)
        * (image_width - 1)
    ).astype(int)

    pixel_y = (
        (np.pi / 2 - lat_grid) / np.pi
        * (image_height - 1)
    ).astype(int)

    sampled = texture[pixel_y, pixel_x].reshape(-1, 3)

    vertex_colors = [
        f"#{r:02X}{g:02X}{b:02X}"
        for r, g, b in sampled
    ]

    triangles_i = []
    triangles_j = []
    triangles_k = []

    for row in range(n_lat - 1):
        for col in range(n_lon):
            next_col = (col + 1) % n_lon

            a = row * n_lon + col
            b = row * n_lon + next_col
            c = (row + 1) * n_lon + col
            d = (row + 1) * n_lon + next_col

            triangles_i.extend((a, b))
            triangles_j.extend((b, d))
            triangles_k.extend((c, c))

    return (
        x.ravel(),
        y.ravel(),
        z.ravel(),
        vertex_colors,
        triangles_i,
        triangles_j,
        triangles_k,
    )


def _add_earth(fig):
    """Add the NASA-textured Earth and subtle atmospheric shell."""
    try:
        mesh = _earth_mesh_data(str(EARTH_TEXTURE))

        fig.add_trace(go.Mesh3d(
            x=mesh[0],
            y=mesh[1],
            z=mesh[2],
            i=mesh[4],
            j=mesh[5],
            k=mesh[6],
            vertexcolor=mesh[3],
            name="NASA Blue Marble Earth",
            hovertemplate="NASA Earth texture<extra></extra>",
            flatshading=False,
            lighting=dict(
                ambient=0.65,
                diffuse=0.8,
                specular=0.35,
                roughness=0.75,
                fresnel=0.15,
            ),
            lightposition=dict(x=100, y=150, z=200),
            showscale=False,
        ))

    except (FileNotFoundError, OSError, ValueError):
        # Keep the scene usable if the texture is unavailable.
        longitude = np.linspace(0, 2 * np.pi, 64)
        latitude = np.linspace(-np.pi / 2, np.pi / 2, 40)
        lon, lat = np.meshgrid(longitude, latitude)

        fig.add_trace(go.Surface(
            x=np.cos(lat) * np.cos(lon),
            y=np.cos(lat) * np.sin(lon),
            z=np.sin(lat),
            colorscale=[
                [0, "#10264A"],
                [1, "#287DA5"],
            ],
            showscale=False,
            name="Earth fallback",
            hoverinfo="skip",
        ))

    # Transparent cyan atmospheric shell.
    longitude = np.linspace(0, 2 * np.pi, 64)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 36)
    lon, lat = np.meshgrid(longitude, latitude)

    radius = 1.035

    fig.add_trace(go.Surface(
        x=radius * np.cos(lat) * np.cos(lon),
        y=radius * np.cos(lat) * np.sin(lon),
        z=radius * np.sin(lat),
        surfacecolor=np.ones_like(lon),
        colorscale=[
            [0, "#38BDF8"],
            [1, "#38BDF8"],
        ],
        opacity=0.10,
        showscale=False,
        name="Atmospheric glow",
        hoverinfo="skip",
        lighting=dict(
            ambient=1,
            diffuse=0,
            specular=0.2,
            roughness=1,
        ),
    ))


def _add_starfield(fig):
    """Add deterministic stars around the orbital scene."""
    rng = np.random.default_rng(2026)

    points = rng.uniform(-2.65, 2.65, size=(260, 3))
    distances = np.linalg.norm(points, axis=1)
    points = points[distances > 2.25]

    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name="Starfield",
        marker=dict(
            size=1.5,
            color="#9FB5D1",
            opacity=0.65,
        ),
        hoverinfo="skip",
        showlegend=False,
    ))

    bright = points[:28]

    fig.add_trace(go.Scatter3d(
        x=bright[:, 0],
        y=bright[:, 1],
        z=bright[:, 2],
        mode="markers",
        name="Bright stars",
        marker=dict(
            size=2.5,
            color="#FFFFFF",
            opacity=0.85,
        ),
        hoverinfo="skip",
        showlegend=False,
    ))


def create_orbital_scene(
    event,
    show_satellites=True,
    show_debris=True,
    show_orbits=True,
    show_labels=True,
):
    """
    Build a textured Earth conjunction scene.

    Visibility arguments are optional to preserve compatibility
    with the existing dashboard integration.
    """
    event_id = str(event.get("event_id", "UNKNOWN EVENT"))
    object_1 = str(event.get("sat_1_id", "OBJECT-1"))
    object_2 = str(event.get("sat_2_id", "OBJECT-2"))

    risk_band = str(event.get("risk_band", "UNKNOWN")).upper()

    risk_colors = {
        "HIGH": "#EF4444",
        "UNCERTAIN": "#F59E0B",
        "LOW": "#10B981",
    }

    risk_color = risk_colors.get(risk_band, MUTED)

    fig = go.Figure()

    # 1. Deep-space background.
    _add_starfield(fig)

    # 2. Real NASA Earth image and atmosphere.
    _add_earth(fig)

    # 3. Selected conjunction objects.
    objects = [
        (object_1, CYAN),
        (object_2, VIOLET),
    ]

    visible_positions = []

    for object_id, color in objects:
        category = _object_category(object_id)

        if not _is_visible(
            category, show_satellites, show_debris
        ):
            continue

        parameters = _stable_parameters(object_id)
        position = _object_position(parameters)

        visible_positions.append((object_id, position))

        if show_orbits:
            orbit = _orbit_coordinates(parameters)

            fig.add_trace(go.Scatter3d(
                x=orbit[0],
                y=orbit[1],
                z=orbit[2],
                mode="lines",
                name=f"{object_id} schematic orbit",
                line=dict(color=color, width=3),
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode=(
                "markers+text"
                if show_labels else "markers"
            ),
            text=[object_id] if show_labels else None,
            textposition="top center",
            textfont=dict(color=WHITE, size=11),
            name=f"{category}: {object_id}",
            marker=dict(
                size=10 if category == "SATELLITE" else 8,
                color=color,
                line=dict(color=WHITE, width=1),
                symbol=(
                    "diamond"
                    if category == "DEBRIS"
                    else "circle"
                ),
            ),
            hovertemplate=(
                f"<b>{object_id}</b>"
                f"<br>Category: {category}"
                "<br>Illustrative position only"
                "<extra></extra>"
            ),
        ))

    # 4. Connector only when both conjunction objects are visible.
    if len(visible_positions) == 2:
        position_1 = visible_positions[0][1]
        position_2 = visible_positions[1][1]

        fig.add_trace(go.Scatter3d(
            x=[position_1[0], position_2[0]],
            y=[position_1[1], position_2[1]],
            z=[position_1[2], position_2[2]],
            mode="lines",
            name="Schematic conjunction connector",
            line=dict(
                color=risk_color,
                width=5,
                dash="dash",
            ),
            hovertemplate=(
                f"<b>{risk_band} risk event</b>"
                "<br>Illustrative connector; not to scale"
                "<extra></extra>"
            ),
        ))

    # 5. Professional dark-space layout.
    fig.update_layout(
        title=dict(
            text=(
                "<b>ORBIT PULSE | 3D ORBITAL INTELLIGENCE</b>"
                f"<br><sup>{event_id} | {risk_band} RISK</sup>"
            ),
            font=dict(color=WHITE, size=16),
            x=0.03,
        ),
        paper_bgcolor=PANEL,
        plot_bgcolor=BACKGROUND,
        font=dict(color=WHITE, family="sans-serif"),
        height=680,
        margin=dict(l=0, r=0, t=85, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(5,11,24,0.75)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
        scene=dict(
            bgcolor=BACKGROUND,
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.15),
            ),
            xaxis=dict(
                visible=False,
                range=[-2.8, 2.8],
            ),
            yaxis=dict(
                visible=False,
                range=[-2.8, 2.8],
            ),
            zaxis=dict(
                visible=False,
                range=[-2.8, 2.8],
            ),
        ),
        uirevision="orbit-pulse-scene",
    )

    return fig
