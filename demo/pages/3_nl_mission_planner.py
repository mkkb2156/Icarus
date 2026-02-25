"""NL Mission Planner — Natural language → cleaning path → 3D visualization.

This is a "vision prototype" showing the final product experience.
It does NOT require Isaac Lab or PSDK — pure Python + Claude API.

Investors can type natural language commands and see 3D cleaning paths generated.
"""

from __future__ import annotations

import json
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mission Planner — Icarus", layout="wide")

st.title("AI Mission Planner")
st.markdown(
    "Natural language → structured task → 3D cleaning path. "
    "Type a command to see the drone's planned trajectory."
)

st.markdown("---")

# ─── Building Configuration ──────────────────────────────────────────

with st.sidebar:
    st.header("Building Configuration")
    building_width = st.slider("Building width (m)", 10, 100, 40)
    building_height = st.slider("Building height (m)", 10, 200, 60)
    building_depth = st.slider("Building depth (m)", 10, 100, 30)
    floor_height = st.slider("Floor height (m)", 2.5, 4.5, 3.0, step=0.5)
    num_floors = int(building_height / floor_height)
    st.caption(f"Total floors: {num_floors}")

    st.markdown("---")
    st.header("Drone Parameters")
    strip_width = st.slider("Cleaning strip width (m)", 0.5, 3.0, 1.5, step=0.5)
    wall_distance = st.slider("Wall distance (m)", 0.5, 3.0, 1.5, step=0.1)
    drone_speed = st.slider("Cruise speed (m/s)", 0.5, 3.0, 1.0, step=0.5)

# ─── Natural Language Input ──────────────────────────────────────────

st.header("Command Input")

example_commands = [
    "清洗東面 3-5 樓外牆",
    "檢查北面 10 樓窗戶周圍裂縫",
    "全棟外牆噴洗，跳過有鷹架的區域",
    "Clean the south facade from floor 1 to 8",
    "Inspect east wall, floors 5 through 12, avoid the balcony areas",
]

selected_example = st.selectbox("Example commands:", ["(Custom input)"] + example_commands)
user_input = st.text_input(
    "Or type your own command:",
    value="" if selected_example == "(Custom input)" else selected_example,
    placeholder="e.g., 清洗東面 3-5 樓外牆",
)

# ─── Task Parsing ────────────────────────────────────────────────────


def parse_command_local(command: str) -> dict:
    """Parse command using simple rules (fallback when API unavailable)."""
    command_lower = command.lower()

    # Detect face
    face = "east"
    for f, keywords in [
        ("east", ["east", "東", "东"]),
        ("west", ["west", "西"]),
        ("south", ["south", "南"]),
        ("north", ["north", "北"]),
    ]:
        if any(k in command_lower for k in keywords):
            face = f
            break

    # Detect floor range
    import re
    floor_min, floor_max = 1, num_floors

    # Match patterns like "3-5", "3 to 5", "3~5", "3至5", "3樓到5樓"
    patterns = [
        r"(\d+)\s*[-~至到]\s*(\d+)",
        r"floor[s]?\s+(\d+)\s+(?:to|through)\s+(\d+)",
        r"(\d+)\s*樓?\s*到\s*(\d+)\s*樓?",
    ]
    for pattern in patterns:
        match = re.search(pattern, command_lower)
        if match:
            floor_min = int(match.group(1))
            floor_max = int(match.group(2))
            break

    # Detect task type
    task_type = "cleaning"
    if any(k in command_lower for k in ["檢查", "inspect", "检查", "crack", "裂縫"]):
        task_type = "inspection"

    # Detect avoid zones
    avoid_zones = []
    if any(k in command_lower for k in ["skip", "avoid", "跳過", "跳过", "鷹架", "balcony"]):
        avoid_zones.append("scaffolding_zone")

    # Detect full building
    if any(k in command_lower for k in ["全棟", "全栋", "entire", "whole", "all"]):
        floor_min, floor_max = 1, num_floors

    floor_min = max(1, min(floor_min, num_floors))
    floor_max = max(floor_min, min(floor_max, num_floors))

    return {
        "task_type": task_type,
        "target_face": face,
        "floor_range": [floor_min, floor_max],
        "avoid_zones": avoid_zones,
        "spray_pressure": "standard" if task_type == "cleaning" else "none",
    }


def parse_command_with_api(command: str) -> dict:
    """Parse command using Claude API with function calling."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        tools = [{
            "name": "plan_drone_mission",
            "description": "Plan a drone facade operation mission based on a natural language command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["cleaning", "inspection"],
                        "description": "Type of facade operation",
                    },
                    "target_face": {
                        "type": "string",
                        "enum": ["east", "west", "south", "north"],
                        "description": "Which building face to operate on",
                    },
                    "floor_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Start and end floor numbers [min, max]",
                    },
                    "avoid_zones": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Areas to avoid (e.g., scaffolding, balconies)",
                    },
                    "spray_pressure": {
                        "type": "string",
                        "enum": ["none", "low", "standard", "high"],
                        "description": "Spray pressure level",
                    },
                },
                "required": ["task_type", "target_face", "floor_range", "spray_pressure"],
            },
        }]

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            tools=tools,
            messages=[{
                "role": "user",
                "content": (
                    f"Parse this drone facade operation command into a structured mission plan. "
                    f"The building has {num_floors} floors. Command: {command}"
                ),
            }],
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.input

    except Exception:
        pass

    return parse_command_local(command)


# ─── Path Generation ─────────────────────────────────────────────────


def generate_boustrophedon_path(
    mission: dict,
    building_w: float,
    building_h: float,
    building_d: float,
    floor_h: float,
    strip_w: float,
    wall_dist: float,
) -> tuple[np.ndarray, dict]:
    """Generate boustrophedon (lawn-mower) cleaning path.

    Returns:
        path: (N, 3) array of waypoints
        stats: mission statistics
    """
    face = mission["target_face"]
    f_min, f_max = mission["floor_range"]

    # Determine wall plane coordinates
    if face == "east":
        wall_x = building_d / 2
        y_range = (-building_w / 2, building_w / 2)
        drone_offset = np.array([wall_dist, 0, 0])
    elif face == "west":
        wall_x = -building_d / 2
        y_range = (-building_w / 2, building_w / 2)
        drone_offset = np.array([-wall_dist, 0, 0])
    elif face == "south":
        wall_x = 0
        y_range = (-building_d / 2, building_d / 2)
        drone_offset = np.array([0, -wall_dist, 0])
    else:  # north
        wall_x = 0
        y_range = (-building_d / 2, building_d / 2)
        drone_offset = np.array([0, wall_dist, 0])

    z_min = (f_min - 1) * floor_h
    z_max = f_max * floor_h

    # Generate boustrophedon path
    waypoints = []
    lateral_range = y_range[1] - y_range[0]
    num_strips = max(1, int(math.ceil(lateral_range / strip_w)))

    # Start from takeoff position
    start_pos = np.array([0, 0, 0])
    waypoints.append(start_pos)

    # Fly to starting position
    if face in ("east", "west"):
        first_wp = np.array([wall_x + drone_offset[0], y_range[0], z_min])
    else:
        first_wp = np.array([y_range[0], wall_x + drone_offset[1], z_min])
    waypoints.append(first_wp)

    # Boustrophedon pattern
    for i in range(num_strips):
        lateral_pos = y_range[0] + (i + 0.5) * strip_w
        lateral_pos = min(lateral_pos, y_range[1])

        if i % 2 == 0:
            # Go up
            z_positions = [z_min, z_max]
        else:
            # Go down
            z_positions = [z_max, z_min]

        for z in z_positions:
            if face in ("east", "west"):
                wp = np.array([wall_x + drone_offset[0], lateral_pos, z])
            else:
                wp = np.array([lateral_pos, wall_x + drone_offset[1], z])
            waypoints.append(wp)

    # Return to start
    waypoints.append(waypoints[-1].copy())
    waypoints[-1][2] = z_max + 5  # fly above
    waypoints.append(start_pos)

    path = np.array(waypoints)

    # Compute statistics
    total_distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    coverage_area = (z_max - z_min) * lateral_range
    flight_time = total_distance / drone_speed
    battery_pct = min(100, (flight_time / (55 * 60)) * 100)  # 55 min max flight
    water_usage = coverage_area * 0.5 if mission["spray_pressure"] != "none" else 0  # L/m²

    stats = {
        "total_distance": total_distance,
        "coverage_area": coverage_area,
        "estimated_time": flight_time,
        "battery_usage": battery_pct,
        "water_usage": water_usage,
        "num_strips": num_strips,
        "num_waypoints": len(waypoints),
    }

    return path, stats


# ─── 3D Visualization ────────────────────────────────────────────────


def create_3d_figure(
    path: np.ndarray, mission: dict, stats: dict,
    bw: float, bh: float, bd: float, fh: float
) -> go.Figure:
    """Create 3D Plotly figure with building wireframe and drone path."""
    fig = go.Figure()

    # Building wireframe
    x_corners = [-bd/2, bd/2, bd/2, -bd/2, -bd/2]
    y_corners = [-bw/2, -bw/2, bw/2, bw/2, -bw/2]

    # Bottom and top rectangles
    for z in [0, bh]:
        fig.add_trace(go.Scatter3d(
            x=x_corners, y=y_corners, z=[z]*5,
            mode="lines", line=dict(color="gray", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # Vertical edges
    for x, y in zip(x_corners[:4], y_corners[:4]):
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[y, y], z=[0, bh],
            mode="lines", line=dict(color="gray", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # Floor lines on target face
    face = mission["target_face"]
    f_min, f_max = mission["floor_range"]
    for floor in range(f_min, f_max + 1):
        z = floor * fh
        if face in ("east", "west"):
            fx = bd/2 if face == "east" else -bd/2
            fig.add_trace(go.Scatter3d(
                x=[fx, fx], y=[-bw/2, bw/2], z=[z, z],
                mode="lines", line=dict(color="lightblue", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            fy = -bw/2 if face == "south" else bw/2
            fig.add_trace(go.Scatter3d(
                x=[-bd/2, bd/2], y=[fy, fy], z=[z, z],
                mode="lines", line=dict(color="lightblue", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

    # Highlight working face
    f_min_z = (f_min - 1) * fh
    f_max_z = f_max * fh
    if face in ("east", "west"):
        fx = bd/2 if face == "east" else -bd/2
        fig.add_trace(go.Mesh3d(
            x=[fx]*4, y=[-bw/2, bw/2, bw/2, -bw/2], z=[f_min_z, f_min_z, f_max_z, f_max_z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="lightblue", opacity=0.2, name="Working Zone",
            showlegend=True,
        ))
    else:
        fy = -bw/2 if face == "south" else bw/2
        fig.add_trace(go.Mesh3d(
            x=[-bd/2, bd/2, bd/2, -bd/2], y=[fy]*4, z=[f_min_z, f_min_z, f_max_z, f_max_z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="lightblue", opacity=0.2, name="Working Zone",
            showlegend=True,
        ))

    # Drone flight path
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2],
        mode="lines+markers",
        line=dict(color="#FF5722", width=4),
        marker=dict(size=3, color="#FF5722"),
        name="Drone Path",
    ))

    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]],
        mode="markers+text", marker=dict(size=8, color="green"),
        text=["START"], textposition="top center",
        name="Start", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]],
        mode="markers+text", marker=dict(size=8, color="red"),
        text=["END"], textposition="top center",
        name="End", showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Mission: {mission['task_type'].title()} — {face.title()} Face, Floors {mission['floor_range'][0]}-{mission['floor_range'][1]}",
    )

    return fig


# ─── Main Logic ───────────────────────────────────────────────────────

if user_input:
    with st.spinner("Parsing command..."):
        # Try API first, fall back to local parsing
        mission = parse_command_local(user_input)  # Default to local for reliability
        if st.sidebar.checkbox("Use Claude API for parsing", value=False):
            mission = parse_command_with_api(user_input)

    # Display parsed mission
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Parsed Mission")
        st.json(mission)

    with col2:
        st.subheader("Mission Summary")
        face_names = {"east": "東面", "west": "西面", "south": "南面", "north": "北面"}
        task_names = {"cleaning": "清洗", "inspection": "檢查"}
        st.markdown(f"""
        - **Task**: {task_names.get(mission['task_type'], mission['task_type'])} ({mission['task_type']})
        - **Target**: {face_names.get(mission['target_face'], mission['target_face'])} ({mission['target_face']})
        - **Floor range**: {mission['floor_range'][0]}F — {mission['floor_range'][1]}F
        - **Spray**: {mission['spray_pressure']}
        - **Avoid zones**: {', '.join(mission['avoid_zones']) if mission['avoid_zones'] else 'None'}
        """)

    st.markdown("---")

    # Generate path
    path, stats = generate_boustrophedon_path(
        mission, building_width, building_height, building_depth,
        floor_height, strip_width, wall_distance,
    )

    # 3D visualization
    st.header("3D Flight Path")
    fig = create_3d_figure(
        path, mission, stats,
        building_width, building_height, building_depth, floor_height,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mission statistics
    st.header("Mission Estimate")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Flight Distance", f"{stats['total_distance']:.0f} m")
    with col2:
        st.metric("Coverage Area", f"{stats['coverage_area']:.0f} m²")
    with col3:
        minutes = stats['estimated_time'] / 60
        st.metric("Est. Flight Time", f"{minutes:.1f} min")
    with col4:
        st.metric("Battery Usage", f"{stats['battery_usage']:.1f}%")
    with col5:
        st.metric("Water Usage", f"{stats['water_usage']:.0f} L")

    st.caption(
        f"Path: {stats['num_strips']} cleaning strips, "
        f"{stats['num_waypoints']} waypoints, "
        f"strip width = {strip_width}m, wall distance = {wall_distance}m"
    )

else:
    st.info("Enter a command above or select an example to generate a mission plan.")
