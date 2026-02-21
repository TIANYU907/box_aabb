"""
viz/scene_viz.py - 3D 场景可视化 (Plotly)

从 v2/examples/panda_planner.py 提取的 3D 可视化函数:
- plot_joint_trajectory: 关节轨迹图
- plot_arm_scene_html: 交互式 3D 场景
- plot_arm_poses_html: 多臂型残影
- create_animation_html: 路径动画
- generate_report: 多方法对比报告
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Joint trajectory
# ═══════════════════════════════════════════════════════════════════════════

def plot_joint_trajectory(waypoints, q_start, q_goal, label="",
                          joint_names=None):
    """画 joint 曲线 → 交互式 plotly HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not waypoints:
        return None
    wps = np.array(waypoints)
    ndim = wps.shape[1]
    fig = make_subplots(
        rows=ndim, cols=1, shared_xaxes=True,
        subplot_titles=[joint_names[d] if joint_names else f"q{d}"
                        for d in range(ndim)],
        vertical_spacing=0.02,
    )
    xs = list(range(len(wps)))
    for d in range(ndim):
        row = d + 1
        fig.add_trace(go.Scatter(
            x=xs, y=wps[:, d].tolist(), mode='lines+markers',
            marker=dict(size=4), line=dict(color='steelblue', width=1.5),
            name=joint_names[d] if joint_names else f"q{d}",
            showlegend=(d == 0),
        ), row=row, col=1)
        fig.add_hline(
            y=float(q_start[d]),
            line=dict(color='green', dash='dash', width=1),
            row=row, col=1)
        fig.add_hline(
            y=float(q_goal[d]),
            line=dict(color='red', dash='dash', width=1),
            row=row, col=1)
    fig.update_layout(
        title=dict(text=f"Joint Trajectory — {label}", font=dict(size=14)),
        height=180 * ndim, width=900,
    )
    fig.update_xaxes(title_text="Waypoint index", row=ndim, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Plotly 3D helpers
# ═══════════════════════════════════════════════════════════════════════════

def _plotly_box_mesh(mn, mx, color='red', opacity=0.40, name='obstacle'):
    """返回一个 plotly Mesh3d 半透明 AABB."""
    import plotly.graph_objects as go
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        hoverinfo='name', flatshading=True,
    )


def _plotly_arm_traces(positions, color='#2196F3', width=5, name='arm',
                       opacity=1.0, showlegend=True):
    """返回 plotly traces (线 + 关节点 + 基座 + 末端)."""
    import plotly.graph_objects as go
    xs = [float(p[0]) for p in positions]
    ys = [float(p[1]) for p in positions]
    zs = [float(p[2]) for p in positions]
    traces = []
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines+markers',
        line=dict(color=color, width=width),
        marker=dict(size=3, color=color, opacity=opacity),
        name=name, opacity=opacity, showlegend=showlegend,
        hoverinfo='name+text',
        text=[f"Link {i}" for i in range(len(xs))],
    ))
    traces.append(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0]], mode='markers',
        marker=dict(size=6, color='black', symbol='square', opacity=opacity),
        name=f'{name} base', showlegend=False,
    ))
    traces.append(go.Scatter3d(
        x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode='markers',
        marker=dict(size=5, color=color, symbol='diamond', opacity=opacity),
        name=f'{name} EE', showlegend=False,
    ))
    return traces


def _plotly_obstacle_traces(scene):
    """返回场景中所有障碍物的 Mesh3d traces."""
    traces = []
    if scene is None:
        return traces
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        if abs(mx[2] - mn[2]) > 100:
            continue
        traces.append(_plotly_box_mesh(mn, mx, color='red', opacity=0.40,
                                       name=obs.name))
    return traces


def _obstacle_corner_pts(scene):
    """收集所有障碍物的 min/max 角点, 用于计算坐标范围."""
    pts = []
    if scene is None:
        return pts
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        if abs(mx[2] - mn[2]) > 100:
            continue
        pts.append(mn)
        pts.append(mx)
    return pts


def _plotly_scene_layout(title="Panda Arm", all_pts=None):
    """返回 3D scene 通用 layout."""
    layout = dict(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=1000, height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if all_pts is not None and len(all_pts) > 0:
        xs = [float(p[0]) for p in all_pts]
        ys = [float(p[1]) for p in all_pts]
        zs = [float(p[2]) for p in all_pts]
        margin = 0.15
        layout['scene']['xaxis'] = dict(
            range=[min(xs) - margin, max(xs) + margin], title='X (m)')
        layout['scene']['yaxis'] = dict(
            range=[min(ys) - margin, max(ys) + margin], title='Y (m)')
        layout['scene']['zaxis'] = dict(
            range=[min(min(zs) - margin, -0.05), max(zs) + margin],
            title='Z (m)')
    return layout


# ═══════════════════════════════════════════════════════════════════════════
# 3D scene visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_arm_scene_html(robot, scene, q_start, q_goal, waypoints=None,
                        title="Panda Arm — Start / Goal"):
    """交互式 3D 场景: 障碍物 + 始末臂型 + 末端轨迹."""
    import plotly.graph_objects as go

    traces = _plotly_obstacle_traces(scene)

    pos_start = robot.get_link_positions(
        np.asarray(q_start, dtype=np.float64))
    pos_goal = robot.get_link_positions(
        np.asarray(q_goal, dtype=np.float64))
    traces += _plotly_arm_traces(pos_start, color='#4CAF50', width=6,
                                 name='Start')
    traces += _plotly_arm_traces(pos_goal, color='#F44336', width=6,
                                 name='Goal')

    all_pts = list(pos_start) + list(pos_goal) + _obstacle_corner_pts(scene)

    if waypoints is not None and len(waypoints) >= 2:
        ee_positions = []
        for q in waypoints:
            pos = robot.get_link_positions(
                np.asarray(q, dtype=np.float64))
            ee_positions.append(pos[-1])
            all_pts.extend(pos)
        ee_x = [float(p[0]) for p in ee_positions]
        ee_y = [float(p[1]) for p in ee_positions]
        ee_z = [float(p[2]) for p in ee_positions]
        traces.append(go.Scatter3d(
            x=ee_x, y=ee_y, z=ee_z, mode='lines+markers',
            line=dict(color='#FF9800', width=4),
            marker=dict(size=2, color='#FF9800'),
            name='EE path', opacity=0.8,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_plotly_scene_layout(title, all_pts))
    return fig


def plot_arm_poses_html(robot, scene, waypoints, n_ghosts=10,
                        title="Panda — Arm Pose Sequence"):
    """交互式 3D 多臂型残影."""
    import plotly.graph_objects as go

    traces = _plotly_obstacle_traces(scene)

    n = len(waypoints)
    if n < 2:
        fig = go.Figure(data=traces)
        fig.update_layout(**_plotly_scene_layout(title))
        return fig

    ghost_idxs = np.linspace(0, n - 1, min(n_ghosts, n), dtype=int)

    import matplotlib.cm as mcm
    colors_rgba = [
        mcm.coolwarm(float(i) / max(len(ghost_idxs) - 1, 1))
        for i in range(len(ghost_idxs))
    ]

    all_pts = _obstacle_corner_pts(scene)
    for k, idx in enumerate(ghost_idxs):
        q = np.asarray(waypoints[idx], dtype=np.float64)
        pos = robot.get_link_positions(q)
        rgba = colors_rgba[k]
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        opacity = 0.3 if 0 < k < len(ghost_idxs) - 1 else 0.95
        width = 3 if 0 < k < len(ghost_idxs) - 1 else 7
        label = f"t={idx}"
        if k == 0:
            label = 'Start'
        elif k == len(ghost_idxs) - 1:
            label = 'Goal'
        traces += _plotly_arm_traces(
            pos, color=hex_color, width=width, name=label,
            opacity=opacity,
            showlegend=(k == 0 or k == len(ghost_idxs) - 1))
        all_pts.extend(pos)

    ee_positions = []
    for q in waypoints:
        pos = robot.get_link_positions(np.asarray(q, dtype=np.float64))
        ee_positions.append(pos[-1])
    ee_x = [float(p[0]) for p in ee_positions]
    ee_y = [float(p[1]) for p in ee_positions]
    ee_z = [float(p[2]) for p in ee_positions]
    traces.append(go.Scatter3d(
        x=ee_x, y=ee_y, z=ee_z, mode='lines',
        line=dict(color='#FF9800', width=5),
        name='EE trajectory', opacity=0.7,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_plotly_scene_layout(title, all_pts))
    return fig


def create_animation_html(robot, scene, waypoints, n_frames=60,
                          title="Panda Path Animation"):
    """交互式 3D 动画 (plotly frames) — 自动循环播放."""
    import plotly.graph_objects as go
    from viz.dynamic_visualizer import resample_path

    smooth_path = resample_path(waypoints, n_frames=n_frames)

    all_link_pos = []
    for q in smooth_path:
        pos = robot.get_link_positions(np.asarray(q, dtype=np.float64))
        all_link_pos.append(pos)

    all_pts = _obstacle_corner_pts(scene)
    for pos in all_link_pos:
        all_pts.extend(pos)

    obs_traces = _plotly_obstacle_traces(scene)
    n_obs = len(obs_traces)

    pos0 = all_link_pos[0]
    arm_xs = [float(p[0]) for p in pos0]
    arm_ys = [float(p[1]) for p in pos0]
    arm_zs = [float(p[2]) for p in pos0]

    arm_trace = go.Scatter3d(
        x=arm_xs, y=arm_ys, z=arm_zs,
        mode='lines+markers',
        line=dict(color='#2196F3', width=7),
        marker=dict(size=4, color='#2196F3'),
        name='Current Arm',
    )
    ee_trace = go.Scatter3d(
        x=[arm_xs[-1]], y=[arm_ys[-1]], z=[arm_zs[-1]],
        mode='lines+markers',
        line=dict(color='#FF9800', width=3),
        marker=dict(size=2, color='#FF9800'),
        name='EE Trail', opacity=0.8,
    )
    base_trace = go.Scatter3d(
        x=[arm_xs[0]], y=[arm_ys[0]], z=[arm_zs[0]],
        mode='markers',
        marker=dict(size=6, color='black', symbol='square'),
        name='Base', showlegend=False,
    )
    ee_marker = go.Scatter3d(
        x=[arm_xs[-1]], y=[arm_ys[-1]], z=[arm_zs[-1]],
        mode='markers',
        marker=dict(size=5, color='#F44336', symbol='diamond'),
        name='EE', showlegend=False,
    )

    data = obs_traces + [arm_trace, ee_trace, base_trace, ee_marker]

    frames = []
    trail_x, trail_y, trail_z = [], [], []
    for i in range(n_frames):
        pos = all_link_pos[i]
        xs = [float(p[0]) for p in pos]
        ys = [float(p[1]) for p in pos]
        zs = [float(p[2]) for p in pos]
        trail_x.append(xs[-1])
        trail_y.append(ys[-1])
        trail_z.append(zs[-1])
        frame_data = [
            go.Scatter3d(x=xs, y=ys, z=zs),
            go.Scatter3d(x=list(trail_x), y=list(trail_y),
                         z=list(trail_z)),
            go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]]),
            go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[zs[-1]]),
        ]
        frames.append(go.Frame(
            data=frame_data,
            traces=list(range(n_obs, n_obs + 4)),
            name=str(i),
        ))

    layout = _plotly_scene_layout(title, all_pts)
    layout['updatemenus'] = [dict(
        type='buttons', showactive=False,
        x=0.05, y=0.05, xanchor='left', yanchor='bottom',
        buttons=[
            dict(label='Play',
                 method='animate',
                 args=[None, dict(
                     frame=dict(duration=50, redraw=True),
                     fromcurrent=True,
                     mode='immediate',
                     transition=dict(duration=0),
                 )]),
            dict(label='Pause',
                 method='animate',
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode='immediate',
                     transition=dict(duration=0),
                 )]),
        ],
    )]
    layout['sliders'] = [dict(
        active=0,
        steps=[dict(
            args=[[str(i)], dict(
                frame=dict(duration=50, redraw=True),
                mode='immediate', transition=dict(duration=0),
            )], label=str(i), method='animate')
            for i in range(n_frames)],
        x=0.05, len=0.9, y=0, xanchor='left',
        currentvalue=dict(prefix='Frame: ', visible=True),
        transition=dict(duration=0),
    )]

    fig = go.Figure(data=data, frames=frames, layout=layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# HTML save
# ═══════════════════════════════════════════════════════════════════════════

def save_plotly_html(fig, filepath):
    """保存 plotly figure 为 HTML."""
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    Path(filepath).write_text(html_str, encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(cfg, scene, results, q_start, q_goal, ndim,
                    out_dir, total_s, prep_info, viz_files=None):
    """生成多方法对比报告."""
    lines = []
    w = lines.append

    w("=" * 70)
    w("  Panda 7-DOF Path Planner — Multi-Method Comparison")
    w("=" * 70)
    w(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Seed       : {cfg.seed}")
    w(f"  DOF        : {ndim}")
    dist = float(np.linalg.norm(q_goal - q_start))
    w(f"  q_start    : {np.array2string(q_start, precision=4)}")
    w(f"  q_goal     : {np.array2string(q_goal, precision=4)}")
    w(f"  config dist: {dist:.4f} rad")
    w("")

    w("--- Scene ---")
    w(f"  obstacles  : {scene.n_obstacles}")
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        w(f"    {obs.name}: center=({(mn[0]+mx[0])/2:.3f}, "
          f"{(mn[1]+mx[1])/2:.3f}, {(mn[2]+mx[2])/2:.3f})  "
          f"size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")
    w("")

    w("--- Config ---")
    w(f"  max_consecutive_miss : {cfg.max_consecutive_miss}")
    w(f"  max_boxes (cap)      : {cfg.max_boxes}")
    w(f"  min_box_size         : {cfg.min_box_size}")
    w(f"  goal_bias            : {cfg.goal_bias}")
    w(f"  guided_sample_ratio  : {cfg.guided_sample_ratio}")
    w(f"  corridor_hops        : {cfg.corridor_hops}")
    w(f"  coarsen_max_rounds   : {cfg.coarsen_max_rounds}")
    w("")

    w("--- Shared Forest ---")
    w(f"  grown boxes      : {prep_info['n_grown']}")
    cs = prep_info.get('coarsen_stats')
    if cs:
        w(f"  after coarsen    : {cs.n_after} "
          f"({cs.n_merges} merges, {cs.n_rounds} rounds)")
    w(f"  AABB cache nodes : {prep_info['n_cache_nodes']}")
    w(f"  grow time        : {prep_info['grow_ms']:8.1f} ms")
    w(f"  cache time       : {prep_info['cache_ms']:8.1f} ms  (parallel)")
    w(f"  coarsen time     : {prep_info['coarsen_ms']:8.1f} ms")
    shared_ms = prep_info['grow_ms'] + prep_info['coarsen_ms']
    w(f"  shared total     : {shared_ms:8.1f} ms  (cache parallel)")
    w("")

    gd = prep_info.get('grow_detail')
    if gd:
        w("--- Grow Forest Breakdown ---")
        w(f"  warmup_fk        : {gd.get('warmup_ms', 0):8.1f} ms")
        w(f"  sample_batch     : {gd['sample_ms']:8.1f} ms  "
          f"({gd['n_sample_calls']} batches)")
        w(f"  is_occupied      : {gd['is_occupied_ms']:8.1f} ms  "
          f"({gd['n_is_occ_calls']} calls)")
        w(f"  can_expand       : {gd.get('probe_ms', 0):8.1f} ms  "
          f"({gd.get('n_probe_calls', 0)} calls, "
          f"{gd.get('n_probe_reject', 0)} rejected)")
        w(f"  find_free_box    : {gd['find_free_box_ms']:8.1f} ms  "
          f"({gd['n_ffb_calls']} calls, {gd['n_ffb_none']} none)")
        w(f"  volume_check     : {gd['volume_check_ms']:8.1f} ms")
        w(f"  add_box          : {gd['add_box_ms']:8.1f} ms  "
          f"({gd['n_absorbed']} absorbed)")
        w(f"  overhead/other   : {gd['overhead_ms']:8.1f} ms")
        w("")

    w("=" * 70)
    w("  METHOD COMPARISON")
    w("=" * 70)
    header = (f"  {'Method':<15s} {'Cost':>8s} {'WP':>4s} "
              f"{'Bridge ms':>10s} {'Plan ms':>10s} {'Total ms':>10s}")
    w(header)
    w("  " + "-" * 65)
    for res in results:
        if res is None:
            continue
        name = res['method']
        ok = res.get('success', False)
        cost_s = f"{res['cost']:.4f}" if ok else "FAIL"
        wp_s = str(len(res['waypoints'])) if ok else "-"
        bridge_ms = res.get('bridge_ms', 0.0)
        plan_ms = res.get('plan_ms', 0.0)
        total_method = bridge_ms + res.get('adj_ms', 0.0) + plan_ms
        w(f"  {name:<15s} {cost_s:>8s} {wp_s:>4s} "
          f"{bridge_ms:>10.0f} {plan_ms:>10.0f} {total_method:>10.0f}")
    w("")

    for res in results:
        if res is None:
            continue
        name = res['method']
        ok = res.get('success', False)
        w(f"--- {name} {'SUCCESS' if ok else 'FAILED'} ---")
        if ok:
            w(f"  cost       : {res['cost']:.4f}")
            w(f"  waypoints  : {len(res['waypoints'])}")
        w(f"  bridge ms  : {res.get('bridge_ms', 0):.1f}")
        w(f"  adj ms     : {res.get('adj_ms', 0):.1f}")
        w(f"  plan ms    : {res.get('plan_ms', 0):.1f}")
        if res.get('bridge_edges'):
            w(f"  bridge edges : {len(res['bridge_edges'])}")
        if res.get('n_before_islands'):
            w(f"  islands    : {res['n_before_islands']} -> "
              f"{res.get('n_after_islands', '?')}")
        w("")

    w("--- Visualization ---")
    if viz_files:
        for name, ms in viz_files:
            w(f"  {name:30s}: {ms:7.0f} ms")
    w("")
    w(f"  Total elapsed      : {total_s:.1f} s")
    w("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    out_dir = Path(out_dir)
    report_path = out_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Report saved -> {report_path}")
    return report_path
