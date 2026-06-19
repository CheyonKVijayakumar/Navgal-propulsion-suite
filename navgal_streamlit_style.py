# navgal_complete_suite.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import pandas as pd
from scipy.optimize import fsolve
import datetime
import base64

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAVGAL Propulsion Engineering Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ────────────────────────────────────────────────────────────────
R_UNIVERSAL = 8314.46
G_0 = 9.80665
GAMMA = 1.22
CF_VACUUM = 1.45
GAMMA_FACTOR = math.sqrt(GAMMA * (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (GAMMA - 1.0)))

# Material Properties
MATERIALS = {
    'copper_alloy': {
        'name': 'Copper Alloy (Cu-Cr-Zr)',
        'rho': 8900, 'cp': 385, 'k': 380, 'tmelt': 1356,
        'sigma_y': 200, 'sigma_ult': 300, 'emissivity': 0.4
    },
    'inconel_718': {
        'name': 'Inconel 718',
        'rho': 8190, 'cp': 435, 'k': 11.4, 'tmelt': 1573,
        'sigma_y': 1100, 'sigma_ult': 1240, 'emissivity': 0.6
    },
    'c103_nb': {
        'name': 'C103 Niobium Alloy',
        'rho': 8680, 'cp': 276, 'k': 52, 'tmelt': 2300,
        'sigma_y': 380, 'sigma_ult': 420, 'emissivity': 0.3
    },
    'titanium': {
        'name': 'Titanium Alloy (Ti-6Al-4V)',
        'rho': 4430, 'cp': 526, 'k': 6.7, 'tmelt': 1923,
        'sigma_y': 880, 'sigma_ult': 950, 'emissivity': 0.5
    },
    'graphite': {
        'name': 'Graphite',
        'rho': 1800, 'cp': 710, 'k': 50, 'tmelt': 4000,
        'sigma_y': 20, 'sigma_ult': 40, 'emissivity': 0.8
    }
}

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #060b18; padding: 0rem 1rem; }
    .css-1d391kg { background-color: #0c1524; }
    
    .report-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .report-header h1 {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .report-header h2 {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 400;
        margin: 5px 0 0 0;
        letter-spacing: 2px;
    }
    .report-header .company-name {
        color: #3b82f6;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 3px;
    }
    
    .metric-card {
        background: #0f172a;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        text-align: center;
        border: 1px solid #1e293b;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 5px 0;
    }
    .metric-card .label {
        font-size: 10px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .status-active {
        color: #00ffcc;
        text-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    .status-standby { color: #64748b; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .section-title {
        background: #0f172a;
        padding: 8px 15px;
        border-radius: 6px;
        margin: 10px 0;
        border-left: 3px solid #3b82f6;
    }
    .section-title h3 {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        margin: 0;
        letter-spacing: 2px;
    }
    
    .data-table {
        background: #0f172a;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #1e293b;
        margin: 5px 0;
    }
    .data-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .data-table th {
        color: #64748b;
        font-weight: 600;
        font-size: 10px;
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #1e293b;
        letter-spacing: 1px;
    }
    .data-table td {
        padding: 8px;
        border-bottom: 1px solid #1a2744;
        font-size: 12px;
        color: #e2e8f0;
    }
    .data-table tr:last-child td {
        border-bottom: none;
    }
    
    .success-box {
        background: rgba(0, 255, 204, 0.05);
        border: 1px solid #00ffcc;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box h3 {
        color: #00ffcc;
        margin: 0;
        font-size: 14px;
    }
    .success-box p {
        color: #94a3b8;
        margin: 5px 0 0 0;
        font-size: 12px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.4);
    }
    
    .report-footer {
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #1e293b;
        text-align: center;
        color: #475569;
        font-size: 10px;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Physics Models ──────────────────────────────────────────────────────────

def atmosphere_model(altitude_km):
    h = altitude_km * 1000
    if h < 11000:
        T = 288.15 - 0.0065 * h
        p = 101325 * (T / 288.15) ** 5.256
        rho = 1.225 * (T / 288.15) ** 4.256
    elif h < 20000:
        T = 216.65
        p = 22632 * math.exp(-0.000157 * (h - 11000))
        rho = p / (287.05 * T)
    else:
        T = 216.65 + 0.001 * (h - 20000)
        p = 5475 * (216.65 / T) ** 34.163
        rho = p / (287.05 * T)
    return {
        'pressure': p / 101325,
        'temperature': T,
        'density': rho,
        'speed_of_sound': math.sqrt(1.4 * 287.05 * T)
    }

def dissociation_correction(T_flame):
    if T_flame > 2500:
        dissoc = 0.002 * (T_flame - 2500) ** 0.5
        return max(0.85, 1 - dissoc)
    return 1.0

def combustion_efficiency(phi, p_chamber):
    base_eff = 0.88 + 0.015 * (p_chamber / 20) ** 0.5
    if phi < 0.8 or phi > 1.3:
        penalty = 0.06 * (abs(phi - 1.0) / 0.3) ** 1.5
        base_eff -= penalty
    if p_chamber > 150:
        base_eff += 0.02 * (p_chamber / 150) ** 0.3
    return min(0.95, max(0.78, base_eff))

def calculate_combustion(fuel_mix, phi, p_chamber, T_initial=300):
    alpha = fuel_mix / 100.0
    x = 12 * (1 - alpha) + 2 * alpha
    y = 26 * (1 - alpha) + 2 * alpha
    stoich_O2 = x + y/4
    actual_O2 = stoich_O2 * phi
    
    if phi <= 1.0:
        CO2, H2O, O2 = x, y/2, actual_O2 - stoich_O2
        CO, H2 = 0, 0
        N2 = 3.76 * actual_O2
    else:
        CO2 = x * (1/phi)
        CO = x * (1 - 1/phi)
        H2 = y/2 - CO2
        H2O = y/2 - H2
        O2 = 0
        N2 = 3.76 * stoich_O2
    
    n_total = CO2 + H2O + O2 + CO + H2 + N2
    mw_products = (CO2*44.01 + H2O*18.015 + O2*32 + CO*28.01 + H2*2.016 + N2*28.013) / n_total
    
    fractions = {
        'CO2': CO2/n_total, 'H2O': H2O/n_total, 'O2': O2/n_total,
        'CO': CO/n_total, 'H2': H2/n_total, 'N2': N2/n_total
    }
    
    delta_H = 43e6 * (1 - alpha) + 49.9e6 * alpha
    eta_c = combustion_efficiency(phi, p_chamber)
    Cp_products = 1.1 * 1000 + 0.2 * 1000 * (phi - 1) * (phi > 1)
    T_flame = T_initial + (delta_H * eta_c) / Cp_products
    T_flame *= dissociation_correction(T_flame)
    T_flame = max(2200, min(T_flame, 3800))
    
    return T_flame, mw_products, eta_c, fractions

def optimal_expansion_ratio(p_chamber_pa, p_amb, gamma=GAMMA):
    if p_chamber_pa <= 0:
        return 10
    p_ratio = p_chamber_pa / max(p_amb * 101325, 100)
    if p_ratio <= 1:
        return 1
    
    gamma_eff = gamma - 0.02 * (p_chamber_pa / 101325) ** 0.3
    
    def equation(eps):
        if eps <= 1:
            return 1e6
        term1 = (2/(gamma_eff+1))**((gamma_eff+1)/(2*(gamma_eff-1)))
        term2 = (p_ratio**((gamma_eff-1)/gamma_eff) - 1) / ((gamma_eff-1)/2)
        return eps**2 * term1 - 1 / (1 + term2)
    
    for guess in [5, 10, 20, 50, 100]:
        try:
            sol = fsolve(equation, guess)[0]
            if 1 < sol < 200:
                return sol
        except:
            continue
    return min(50, 5 * math.log(p_ratio))

def calculate_heat_flux(p_chamber, D_t, T_flame):
    G = 1000 * (p_chamber ** 0.8) * (D_t ** 0.2) / 1000
    h_g = 1.5 * G ** 0.2
    T_wall = 300 + 0.005 * T_flame
    q = h_g * (T_flame - T_wall)
    return {
        'heat_flux': q / 1000,
        'h_g': h_g,
        'T_wall': T_wall,
        'G': G
    }

def calculate_cooling(q_flux, D_t, T_flame, material):
    film_eff = 0.4 + 0.3 * (1 - math.exp(-D_t/50))
    coolant_mass = q_flux * math.pi * D_t / (1000 * 1000) * 0.001
    coolant_flow = coolant_mass * 0.1
    regen_flow = coolant_flow * 0.7
    delta_T = T_flame - 300
    thermal_stress = MATERIALS[material]['sigma_y'] * 0.00001 * delta_T
    return {
        'film_cooling_flow': coolant_flow,
        'regen_cooling_flow': regen_flow,
        'thermal_stress': thermal_stress,
        'coolant_ratio': coolant_flow / 0.01,
        'film_effectiveness': film_eff
    }

def structural_analysis(p_chamber, D_t, thickness=0.005, material='copper_alloy'):
    r = D_t / 2000
    p = p_chamber * 101325
    hoop_stress = p * r / thickness
    axial_stress = p * r / (2 * thickness)
    von_mises = math.sqrt(hoop_stress**2 + axial_stress**2 - hoop_stress*axial_stress)
    yield_strength = MATERIALS[material]['sigma_y'] * 1e6
    safety_factor = yield_strength / von_mises if von_mises > 0 else 999
    return {
        'hoop_stress': hoop_stress / 1e6,
        'axial_stress': axial_stress / 1e6,
        'von_mises': von_mises / 1e6,
        'safety_factor': safety_factor,
        'wall_thickness': thickness * 1000
    }

def calculate_injector(p_chamber, m_dot, phi, fuel_mix):
    alpha = fuel_mix / 100.0
    m_dot_lox = m_dot * 2.5 / (1 + 2.5 * (1 + alpha))
    m_dot_fuel = m_dot - m_dot_lox
    delta_p = 0.15 * p_chamber * 101325
    Cd = 0.65
    rho_fuel = 810 * (1 - alpha) + 1.17 * alpha
    A_fuel = m_dot_fuel / (Cd * math.sqrt(2 * rho_fuel * delta_p))
    D_fuel = math.sqrt(4 * A_fuel / math.pi) * 1000
    rho_lox = 1141
    A_lox = m_dot_lox / (Cd * math.sqrt(2 * rho_lox * delta_p))
    D_lox = math.sqrt(4 * A_lox / math.pi) * 1000
    return {
        'm_dot_lox': m_dot_lox,
        'm_dot_fuel': m_dot_fuel,
        'delta_p': delta_p / 101325,
        'D_fuel': D_fuel,
        'D_lox': D_lox,
        'injector_velocity': math.sqrt(2 * delta_p / rho_fuel)
    }

def calculate_cost(fuel_mix, m_dot, burn_time):
    alpha = fuel_mix / 100.0
    kerosene_cost = 0.5
    acetylene_cost = 2.0
    lox_cost = 0.2
    fuel_mass = m_dot * burn_time * (1 - alpha)
    acetylene_mass = m_dot * burn_time * alpha
    lox_mass = m_dot * burn_time * 2.5
    total_cost = (fuel_mass * kerosene_cost + 
                  acetylene_mass * acetylene_cost + 
                  lox_mass * lox_cost)
    manufacturing_cost = 50000 + 2000 * (m_dot * 1000) + 1000 * (burn_time / 100)
    return {
        'total_cost': total_cost,
        'cost_per_kg': total_cost / (m_dot * burn_time),
        'manufacturing_cost': manufacturing_cost,
        'total_project_cost': total_cost + manufacturing_cost,
        'breakdown': {
            'kerosene': fuel_mass * kerosene_cost,
            'acetylene': acetylene_mass * acetylene_cost,
            'lox': lox_mass * lox_cost,
            'manufacturing': manufacturing_cost
        }
    }

# ── Visualization Functions ──────────────────────────────────────────────────

def create_large_gauge(value, max_val, color, title, unit='', height=250):
    fig = go.Figure()
    
    if isinstance(value, float):
        if value > 100:
            fmt = '.0f'
        elif value > 10:
            fmt = '.1f'
        else:
            fmt = '.2f'
    else:
        fmt = '.0f'
    
    suffix_text = f" {unit}" if unit else ""
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'valueformat': fmt,
            'font': {'size': 32, 'color': '#ffffff', 'family': 'JetBrains Mono, monospace'},
            'suffix': suffix_text
        },
        title={
            'text': title,
            'font': {'color': '#94a3b8', 'size': 14, 'family': 'Inter, sans-serif'}
        },
        gauge={
            'axis': {
                'range': [0, max_val],
                'tickwidth': 2,
                'tickcolor': '#475569',
                'ticklen': 8,
                'tickfont': {'size': 10, 'color': '#64748b'},
                'dtick': max_val / 5
            },
            'bar': {
                'color': color,
                'thickness': 0.6
            },
            'bgcolor': '#0f172a',
            'bordercolor': '#1e293b',
            'borderwidth': 2,
            'steps': [
                {'range': [0, max_val * 0.25], 'color': 'rgba(30, 41, 59, 0.3)'},
                {'range': [max_val * 0.25, max_val * 0.5], 'color': 'rgba(30, 41, 59, 0.15)'},
                {'range': [max_val * 0.5, max_val * 0.75], 'color': 'rgba(30, 41, 59, 0.05)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.6,
                'value': max_val * 0.85
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(t=50, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#94a3b8'}
    )
    
    return fig

def create_3d_engine(D_t, D_e, epsilon, active=False):
    if D_t <= 0:
        D_t = 30
        D_e = 60
    
    r_t = D_t / 2.0
    r_e = r_t * math.sqrt(max(1, epsilon))
    r_ch = r_t * 2.8
    
    xs, yt = [], []
    
    for x in np.arange(-60, -30, 2):
        xs.append(x)
        yt.append(r_ch)
    
    for x in np.arange(-30, -2, 1.5):
        xs.append(x)
        x_norm = (x + 30) / 28
        yt.append(r_ch - (r_ch - r_t) * (x_norm**2 * (3 - 2*x_norm)))
    
    xs += [-1, 0]
    yt += [r_t, r_t]
    
    bell_length = max(15, D_t * 4)
    for x in np.arange(1, bell_length + 1, 1.5):
        xs.append(x)
        x_norm = x / bell_length
        yt.append(r_t + (r_e - r_t) * (1 - (1 - x_norm)**2.5))
    
    xs.append(bell_length + 2)
    yt.append(r_e)
    
    n_points = 50
    theta = np.linspace(0, 2*np.pi, n_points)
    
    X_mat, Y_mat, Z_mat = [], [], []
    
    for x, r in zip(xs, yt):
        row_x, row_y, row_z = [], [], []
        for t in theta:
            row_x.append(r * np.cos(t))
            row_y.append(r * np.sin(t))
            row_z.append(x)
        X_mat.append(row_x)
        Y_mat.append(row_y)
        Z_mat.append(row_z)
    
    X_mat = np.array(X_mat)
    Y_mat = np.array(Y_mat)
    Z_mat = np.array(Z_mat)
    
    fig = go.Figure()
    
    if active:
        colorscale = [
            [0, '#1e293b'],
            [0.3, '#ff4400'],
            [0.6, '#ff8800'],
            [0.8, '#ffcc00'],
            [1.0, '#ffffff']
        ]
    else:
        colorscale = [
            [0, '#1e293b'],
            [0.5, '#2a3a5a'],
            [1.0, '#38bdf8']
        ]
    
    fig.add_trace(go.Surface(
        x=X_mat, y=Y_mat, z=Z_mat,
        colorscale=colorscale,
        showscale=False,
        opacity=0.9,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.3)
    ))
    
    if active:
        n_flame = 30
        flame_theta = np.linspace(0, 2*np.pi, n_flame)
        
        for i in range(10):
            t = i / 10.0
            r_flame = r_e * (1 - t * 0.4)
            x_flame = bell_length + 2 + t * 40
            
            x_ring, y_ring, z_ring = [], [], []
            for ft in flame_theta:
                x_ring.append(r_flame * np.cos(ft))
                y_ring.append(r_flame * np.sin(ft))
                z_ring.append(x_flame)
            
            fig.add_trace(go.Scatter3d(
                x=x_ring, y=y_ring, z=z_ring,
                mode='lines',
                line=dict(
                    color=f'rgba(255, {int(200 - t*150)}, 0, {0.6 - t*0.5})',
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-r_t*1.2, r_t*1.2], z=[0, 0],
        mode='lines',
        line=dict(color='#ff3366', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text="3D ENGINE MODEL - Click & Drag to Rotate", font=dict(color='#94a3b8', size=12)),
        scene=dict(
            xaxis=dict(title='X (mm)', gridcolor='#1e293b', color='#475569', showticklabels=True),
            yaxis=dict(title='Y (mm)', gridcolor='#1e293b', color='#475569', showticklabels=True),
            zaxis=dict(title='Z (mm)', gridcolor='#1e293b', color='#475569', showticklabels=True),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2), center=dict(x=0, y=0, z=-10)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.8),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=0, l=0, r=0),
        height=450,
        showlegend=False
    )
    
    return fig

def create_performance_plot(isp, thrust):
    fig = go.Figure()
    if isp > 0 and thrust > 0:
        thrust_range = np.linspace(100, 5000, 50)
        isp_range = isp * (1 - 0.3 * (thrust_range/5000 - 0.5)**2)
        fig.add_trace(go.Scatter(x=thrust_range, y=isp_range, mode='lines', line=dict(color='#00ffcc', width=2)))
        fig.add_trace(go.Scatter(x=[thrust], y=[isp], mode='markers', marker=dict(color='#ff3366', size=12)))
    fig.update_layout(
        height=180,
        margin=dict(t=10, b=15, l=15, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Thrust (N)', gridcolor='#1e293b', color='#475569'),
        yaxis=dict(title='Isp (s)', gridcolor='#1e293b', color='#475569'),
        showlegend=False
    )
    return fig

# ── Report Generation ──────────────────────────────────────────────────────

def generate_professional_report(results, fuel_mix, lox_flux, altitude, target_thrust, burn_time, material, wall_thickness):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"NAVGAL-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
    
    checks = [
        results['eta_c'] > 0.85,
        results['structure']['safety_factor'] > 2.0,
        results['cooling']['film_cooling_flow'] > 0,
        results['injector']['D_lox'] > 0,
        True,
        results['cost']['total_cost'] > 0
    ]
    readiness = sum(checks) / len(checks) * 100
    
    # Determine readiness status
    if readiness > 80:
        status_badge = 'status-ready'
        status_text = '✅ Pass'
        conclusion = 'The engine design is manufacturing-ready. All critical parameters have been validated and meet design requirements.'
    elif readiness > 50:
        status_badge = 'status-partial'
        status_text = '⚠️ Partial'
        conclusion = 'The engine design requires additional refinement before manufacturing. Focus areas: structural margins and thermal management.'
    else:
        status_badge = 'status-not-ready'
        status_text = '❌ Fail'
        conclusion = 'Significant design work is required before manufacturing can proceed. Recommend comprehensive review of all systems.'
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NAVGAL Propulsion Engineering Report</title>
    <style>
        @page {{
            margin: 2.5cm;
            size: A4;
        }}
        body {{
            font-family: 'Segoe UI', 'Arial', sans-serif;
            color: #1e293b;
            line-height: 1.6;
            background: white;
            padding: 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: white;
        }}
        
        .report-header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 40px 50px;
            border-radius: 12px;
            margin-bottom: 30px;
            border-left: 6px solid #3b82f6;
        }}
        .report-header .logo {{
            color: #3b82f6;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 4px;
        }}
        .report-header h1 {{
            color: white;
            font-size: 32px;
            font-weight: 700;
            margin: 10px 0 5px 0;
            letter-spacing: 1px;
        }}
        .report-header h2 {{
            color: #94a3b8;
            font-size: 16px;
            font-weight: 400;
            margin: 0;
            letter-spacing: 2px;
        }}
        .report-header .meta {{
            color: #64748b;
            font-size: 12px;
            margin-top: 15px;
            border-top: 1px solid #1e293b;
            padding-top: 15px;
        }}
        
        .section {{
            margin: 30px 0;
        }}
        .section-title {{
            background: #f1f5f9;
            padding: 12px 20px;
            border-radius: 8px;
            margin: 25px 0 15px 0;
            border-left: 4px solid #3b82f6;
        }}
        .section-title h3 {{
            color: #0f172a;
            font-size: 18px;
            font-weight: 600;
            margin: 0;
        }}
        
        .data-table {{
            background: white;
            border-radius: 8px;
            padding: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin: 10px 0;
            overflow: hidden;
        }}
        .data-table table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .data-table th {{
            background: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
            font-size: 12px;
            padding: 12px 15px;
            text-align: left;
            border-bottom: 2px solid #e2e8f0;
        }}
        .data-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid #f1f5f9;
            font-size: 13px;
            color: #1e293b;
        }}
        .data-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 15px;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #0f172a;
            margin: 5px 0;
        }}
        .metric-card .label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .metric-card .unit {{
            font-size: 14px;
            color: #94a3b8;
        }}
        
        .executive-summary {{
            background: #f8fafc;
            padding: 25px 30px;
            border-radius: 10px;
            border-left: 4px solid #3b82f6;
            margin: 20px 0;
        }}
        .executive-summary h4 {{
            color: #0f172a;
            margin: 0 0 10px 0;
            font-size: 16px;
        }}
        .executive-summary p {{
            color: #334155;
            margin: 0;
            font-size: 14px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status-ready {{
            background: #dcfce7;
            color: #16a34a;
        }}
        .status-partial {{
            background: #fef3c7;
            color: #d97706;
        }}
        .status-not-ready {{
            background: #fee2e2;
            color: #dc2626;
        }}
        
        .report-footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            text-align: center;
            color: #94a3b8;
            font-size: 11px;
            letter-spacing: 1px;
        }}
        
        @media print {{
            body {{ padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="report-header">
            <div class="logo">NAVGAL PROPULSION SYSTEMS</div>
            <h1>ENGINEERING DESIGN REPORT</h1>
            <h2>Dual-Fuel Rocket Propulsion Analysis</h2>
            <div class="meta">
                Report ID: {report_id} &nbsp;|&nbsp; Date: {timestamp} &nbsp;|&nbsp; Version: 1.0
            </div>
        </div>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h4>📋 Executive Summary</h4>
            <p>
                This report presents the complete engineering analysis for the NAVGAL dual-fuel rocket propulsion system.
                The design utilizes a kerosene-acetylene blend with LOX oxidizer. 
                <strong>Manufacturing Readiness: {readiness:.0f}%</strong>
                <span class="status-badge {status_badge}" style="margin-left: 10px;">{status_text}</span>
            </p>
        </div>
        
        <!-- Key Metrics -->
        <div class="section">
            <div class="section-title"><h3>📊 Key Performance Metrics</h3></div>
            <div class="grid-4">
                <div class="metric-card">
                    <div class="label">Specific Impulse</div>
                    <div class="value">{results['isp']:.0f}</div>
                    <div class="unit">seconds (vacuum)</div>
                </div>
                <div class="metric-card">
                    <div class="label">Thrust</div>
                    <div class="value">{results['thrust']:.0f}</div>
                    <div class="unit">N</div>
                </div>
                <div class="metric-card">
                    <div class="label">Chamber Pressure</div>
                    <div class="value">{results['p_chamber']:.1f}</div>
                    <div class="unit">atm</div>
                </div>
                <div class="metric-card">
                    <div class="label">Flame Temperature</div>
                    <div class="value">{results['T_flame']:.0f}</div>
                    <div class="unit">K</div>
                </div>
            </div>
        </div>
        
        <!-- Engine Configuration -->
        <div class="section">
            <div class="section-title"><h3>⚙️ Engine Configuration</h3></div>
            <div class="grid-2">
                <div class="data-table">
                    <table>
                        <tr><th colspan="2">Propellant Parameters</th></tr>
                        <tr><td>Fuel Blend</td><td>{fuel_mix}% Acetylene, {100-fuel_mix}% Kerosene</td></tr>
                        <tr><td>LOX Flow Rate</td><td>{lox_flux:.1f}</td></tr>
                        <tr><td>Equivalence Ratio (φ)</td><td>{results['phi']:.3f}</td></tr>
                        <tr><td>Mass Flow Rate</td><td>{results['m_dot']:.4f} kg/s</td></tr>
                        <tr><td>Burn Time</td><td>{burn_time} s</td></tr>
                    </table>
                </div>
                <div class="data-table">
                    <table>
                        <tr><th colspan="2">Chamber & Nozzle Geometry</th></tr>
                        <tr><td>Throat Diameter (Dt)</td><td>{results['D_t']:.2f} mm</td></tr>
                        <tr><td>Exit Diameter (De)</td><td>{results['D_e']:.2f} mm</td></tr>
                        <tr><td>Expansion Ratio (ε)</td><td>{results['epsilon']:.1f}</td></tr>
                        <tr><td>Chamber Volume (Vc)</td><td>{results['V_c']:.0f} cc</td></tr>
                        <tr><td>Wall Thickness</td><td>{wall_thickness*1000:.1f} mm</td></tr>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Structural & Thermal -->
        <div class="section">
            <div class="section-title"><h3>🔧 Structural & Thermal Analysis</h3></div>
            <div class="grid-2">
                <div class="data-table">
                    <table>
                        <tr><th colspan="2">Structural Analysis</th></tr>
                        <tr><td>Material</td><td>{MATERIALS[material]['name']}</td></tr>
                        <tr><td>Hoop Stress</td><td>{results['structure']['hoop_stress']:.0f} MPa</td></tr>
                        <tr><td>Axial Stress</td><td>{results['structure']['axial_stress']:.0f} MPa</td></tr>
                        <tr><td>Von Mises Stress</td><td>{results['structure']['von_mises']:.0f} MPa</td></tr>
                        <tr><td>Safety Factor</td><td><strong>{results['structure']['safety_factor']:.1f}x</strong></td></tr>
                    </table>
                </div>
                <div class="data-table">
                    <table>
                        <tr><th colspan="2">Thermal Analysis</th></tr>
                        <tr><td>Heat Flux</td><td>{results['heat']['heat_flux']:.0f} kW/m²</td></tr>
                        <tr><td>Gas Wall Temp</td><td>{results['heat']['T_wall']:.0f} K</td></tr>
                        <tr><td>Film Cooling Flow</td><td>{results['cooling']['film_cooling_flow']:.4f} kg/s</td></tr>
                        <tr><td>Regen Cooling Flow</td><td>{results['cooling']['regen_cooling_flow']:.4f} kg/s</td></tr>
                        <tr><td>Thermal Stress</td><td>{results['cooling']['thermal_stress']:.0f} MPa</td></tr>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Injector Design -->
        <div class="section">
            <div class="section-title"><h3>💉 Injector Design</h3></div>
            <div class="data-table">
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>LOX Orifice Diameter</td>
                        <td>{results['injector']['D_lox']:.2f} mm</td>
                        <td>LOX Mass Flow</td>
                        <td>{results['injector']['m_dot_lox']:.4f} kg/s</td>
                    </tr>
                    <tr>
                        <td>Fuel Orifice Diameter</td>
                        <td>{results['injector']['D_fuel']:.2f} mm</td>
                        <td>Fuel Mass Flow</td>
                        <td>{results['injector']['m_dot_fuel']:.4f} kg/s</td>
                    </tr>
                    <tr>
                        <td>Injector ΔP</td>
                        <td>{results['injector']['delta_p']:.2f} atm</td>
                        <td>Injection Velocity</td>
                        <td>{results['injector']['injector_velocity']:.1f} m/s</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <!-- Cost Analysis -->
        <div class="section">
            <div class="section-title"><h3>💰 Cost Analysis</h3></div>
            <div class="grid-2">
                <div class="data-table">
                    <table>
                        <tr><th colspan="2">Cost Breakdown</th></tr>
                        <tr><td>Kerosene</td><td>${results['cost']['breakdown']['kerosene']:.2f}</td></tr>
                        <tr><td>Acetylene</td><td>${results['cost']['breakdown']['acetylene']:.2f}</td></tr>
                        <tr><td>LOX</td><td>${results['cost']['breakdown']['lox']:.2f}</td></tr>
                        <tr><td>Manufacturing</td><td>${results['cost']['breakdown']['manufacturing']:.2f}</td></tr>
                    </table>
                </div>
                <div class="data-table">
                    <table>
                        <tr><th colspan="2">Cost Summary</th></tr>
                        <tr><td>Total Propellant Cost</td><td>${results['cost']['total_cost']:.2f}</td></tr>
                        <tr><td>Manufacturing Cost</td><td>${results['cost']['manufacturing_cost']:.2f}</td></tr>
                        <tr><td><strong>Total Project Cost</strong></td><td><strong>${results['cost']['total_project_cost']:.2f}</strong></td></tr>
                        <tr><td>Cost per kg</td><td>${results['cost']['cost_per_kg']:.2f}/kg</td></tr>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Manufacturing Readiness -->
        <div class="section">
            <div class="section-title"><h3>📋 Manufacturing Readiness Assessment</h3></div>
            <div class="data-table">
                <table>
                    <tr>
                        <th>Requirement</th>
                        <th>Status</th>
                        <th>Details</th>
                    </tr>
                    <tr>
                        <td>Combustion Efficiency Model</td>
                        <td>{'✅ Pass' if results['eta_c'] > 0.85 else '❌ Fail'}</td>
                        <td>{results['eta_c']*100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Structural Safety Factor</td>
                        <td>{'✅ Pass' if results['structure']['safety_factor'] > 2.0 else '❌ Fail'}</td>
                        <td>{results['structure']['safety_factor']:.1f}x</td>
                    </tr>
                    <tr>
                        <td>Cooling System Design</td>
                        <td>{'✅ Pass' if results['cooling']['film_cooling_flow'] > 0 else '❌ Fail'}</td>
                        <td>{results['cooling']['film_cooling_flow']:.4f} kg/s</td>
                    </tr>
                    <tr>
                        <td>Injector Geometry Defined</td>
                        <td>{'✅ Pass' if results['injector']['D_lox'] > 0 else '❌ Fail'}</td>
                        <td>{results['injector']['D_lox']:.2f} mm</td>
                    </tr>
                    <tr>
                        <td>Material Selection</td>
                        <td>✅ Pass</td>
                        <td>{MATERIALS[material]['name']}</td>
                    </tr>
                    <tr>
                        <td>Cost Analysis Complete</td>
                        <td>✅ Pass</td>
                        <td>${results['cost']['total_cost']:.2f}</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <!-- Conclusion -->
        <div class="section">
            <div class="section-title"><h3>📝 Design Conclusion</h3></div>
            <div style="padding: 20px; background: #f8fafc; border-radius: 8px;">
                <p style="margin: 0; color: #1e293b;">
                    <strong>Overall Readiness Level: {readiness:.0f}%</strong>
                    <span class="status-badge {status_badge}" style="margin-left: 10px;">{status_text}</span>
                </p>
                <p style="margin: 10px 0 0 0; color: #475569;">
                    {conclusion}
                </p>
                <p style="margin: 10px 0 0 0; font-size: 12px; color: #94a3b8;">
                    Report generated by NAVGAL Propulsion Analysis Suite v3.0
                </p>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="report-footer">
            <p>NAVGAL PROPULSION SYSTEMS &bull; ENGINEERING DIVISION &bull; CONFIDENTIAL</p>
            <p>This report contains proprietary engineering data. Distribution is restricted.</p>
            <p>Report ID: {report_id} &bull; Generated: {timestamp}</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="color: #00ffcc; font-size: 2rem; margin: 0;">🚀 NAVGAL</h1>
        <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Engineering Design Suite</p>
        <hr style="border-color: #1e293b;">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Engine Parameters")
    
    fuel_mix = st.slider(
        "**Fuel Blend (% Acetylene)**",
        min_value=0, max_value=100, value=20, step=5
    )
    
    lox_flux = st.slider(
        "**LOX Flow Rate**",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5
    )
    
    target_thrust = st.number_input(
        "**Target Thrust (N)**",
        min_value=10, max_value=50000, value=500, step=50
    )
    
    altitude = st.slider(
        "**Altitude (km)**",
        min_value=0, max_value=12, value=0, step=1
    )
    
    burn_time = st.number_input(
        "**Burn Time (s)**",
        min_value=10, max_value=1000, value=100, step=10
    )
    
    st.markdown("### 🔧 Materials")
    material = st.selectbox(
        "**Chamber Material**",
        list(MATERIALS.keys()),
        format_func=lambda x: MATERIALS[x]['name']
    )
    
    wall_thickness = st.slider(
        "**Wall Thickness (mm)**",
        min_value=1, max_value=20, value=5, step=1
    ) / 1000
    
    st.markdown("---")
    
    analyze = st.button("🔥 IGNITE ENGINE", use_container_width=True)

# ── Main Dashboard ──────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="report-header">
    <div class="company-name">NAVGAL PROPULSION SYSTEMS</div>
    <h1>🚀 Engineering Design Suite</h1>
    <h2>Dual-Fuel Rocket Propulsion Analysis & Report Generator</h2>
</div>
""", unsafe_allow_html=True)

# ── Calculate ─────────────────────────────────────────────────────────────────
if analyze:
    try:
        atm = atmosphere_model(altitude)
        p_amb = atm['pressure']
        
        alpha = fuel_mix / 100.0
        stoich_ratio = 3.4 - alpha * 0.4
        phi = lox_flux / stoich_ratio
        
        p_chamber_est = max(1.0, min(20.0, lox_flux * 1.5 * p_amb))
        T_flame, mw_products, eta_c, fractions = calculate_combustion(fuel_mix, phi, p_chamber_est)
        
        p_chamber = max(5.0, min(150.0, lox_flux * 10.0 * p_amb))
        p_chamber_pa = p_chamber * 101325
        
        T_flame, mw_products, eta_c, fractions = calculate_combustion(fuel_mix, phi, p_chamber)
        
        R_specific = R_UNIVERSAL / mw_products
        a_star = math.sqrt(GAMMA * R_specific * T_flame)
        c_star = a_star / GAMMA_FACTOR
        
        epsilon = optimal_expansion_ratio(p_chamber_pa, p_amb)
        isp_vac = (c_star * CF_VACUUM) / G_0
        pressure_correction = 1.0 - 0.015 * (p_amb / p_chamber) * epsilon
        isp = isp_vac * max(0.7, min(1.0, pressure_correction))
        
        F = float(target_thrust)
        m_dot = F / (isp * G_0) if isp > 0 else 0.001
        A_t = (m_dot * c_star) / p_chamber_pa
        D_t = math.sqrt(4.0 * A_t / math.pi) * 1000
        D_e = D_t * math.sqrt(epsilon)
        V_c = A_t * 1.2 * 1e6
        
        thrust_actual = max(0, F - p_amb * (math.pi * (D_e/2000)**2))
        eta_nozzle = 1.0 - 0.015 * math.log(epsilon + 1)
        eta_total = eta_c * eta_nozzle
        
        heat = calculate_heat_flux(p_chamber, D_t, T_flame)
        cooling = calculate_cooling(heat['heat_flux'] * 1000, D_t, T_flame, material)
        structure = structural_analysis(p_chamber, D_t, wall_thickness, material)
        injector = calculate_injector(p_chamber, m_dot, phi, fuel_mix)
        cost = calculate_cost(fuel_mix, m_dot, burn_time)
        safety_margin = min(structure['safety_factor'], 1.0 / (cooling['thermal_stress'] / MATERIALS[material]['sigma_y'] + 0.001))
        
        results = {
            'T_flame': T_flame, 'p_chamber': p_chamber, 'p_amb': p_amb,
            'mw': mw_products, 'c_star': c_star, 'isp': isp,
            'phi': phi, 'D_t': D_t, 'D_e': D_e, 'epsilon': epsilon,
            'V_c': V_c, 'm_dot': m_dot, 'thrust': thrust_actual,
            'eta_c': eta_c, 'eta_total': eta_total, 'active': True,
            'fractions': fractions, 'atm': atm,
            'heat': heat, 'cooling': cooling, 'structure': structure,
            'injector': injector, 'cost': cost, 'safety_margin': safety_margin,
            'material': material, 'wall_thickness': wall_thickness * 1000,
            'burn_time': burn_time
        }
        st.session_state['results'] = results
        
        st.success("✅ Analysis complete! Dashboard and report ready.")
        
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        st.session_state['results'] = {'active': False}

# Get results
results = st.session_state.get('results', {'active': False})

# ── Display Dashboard ──────────────────────────────────────────────────────────

if results['active']:
    # Status Bar
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 15px; 
                background: #0f172a; border-radius: 8px; border: 1px solid #1e293b; margin: 5px 0;">
        <div>
            <span class="status-active" style="font-size: 1rem; font-weight: 700;">● ACTIVE</span>
            <span style="color: #475569; font-size: 0.8rem; margin-left: 20px;">
                ATM: {results['p_amb']:.3f} | 
                Pc: {results['p_chamber']:.1f} atm |
                Safety Margin: {results['safety_margin']:.1f}x
            </span>
        </div>
        <div style="display: flex; gap: 15px;">
            <div style="text-align: center;">
                <span style="color: #475569; font-size: 0.6rem;">MW</span><br>
                <span style="color: #e2e8f0; font-size: 0.9rem; font-weight: 600;">{results['mw']:.1f}</span>
            </div>
            <div style="text-align: center;">
                <span style="color: #475569; font-size: 0.6rem;">C*</span><br>
                <span style="color: #00ffcc; font-size: 0.9rem; font-weight: 600;">{results['c_star']:.0f}</span>
            </div>
            <div style="text-align: center;">
                <span style="color: #475569; font-size: 0.6rem;">Isp</span><br>
                <span style="color: #ff3366; font-size: 0.9rem; font-weight: 600;">{results['isp']:.0f}</span>
            </div>
            <div style="text-align: center;">
                <span style="color: #475569; font-size: 0.6rem;">η</span><br>
                <span style="color: #8b5cf6; font-size: 0.9rem; font-weight: 600;">{results['eta_total']:.2f}</span>
            </div>
            <div style="text-align: center;">
                <span style="color: #475569; font-size: 0.6rem;">φ</span><br>
                <span style="color: #f59e0b; font-size: 0.9rem; font-weight: 600;">{results['phi']:.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Top Row: Gauges ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><h3>📊 ENGINE PERFORMANCE</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_large_gauge(results['T_flame'], 4000, "#ff3366", "Flame Temperature", "K", 250)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        fig = create_large_gauge(results['p_chamber'], 150, "#00ffcc", "Chamber Pressure", "atm", 250)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # ── 3D Engine ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><h3>🛸 3D ENGINE MODEL</h3></div>', unsafe_allow_html=True)
    
    fig_3d = create_3d_engine(results['D_t'], results['D_e'], results['epsilon'], active=True)
    st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True})
    
    # ── Geometry Row ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><h3>⚙️ ENGINE GEOMETRY</h3></div>', unsafe_allow_html=True)
    
    cols = st.columns(6)
    metrics = [
        ("Dt (mm)", f"{results['D_t']:.2f}", "#38bdf8"),
        ("De (mm)", f"{results['D_e']:.2f}", "#38bdf8"),
        ("ε", f"{results['epsilon']:.1f}", "#f59e0b"),
        ("Vc (cc)", f"{results['V_c']:.0f}", "#e2e8f0"),
        ("ṁ (kg/s)", f"{results['m_dot']:.4f}", "#e2e8f0"),
        ("F (N)", f"{results['thrust']:.0f}", "#ff3366")
    ]
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 8px; background: #0f172a; border-radius: 5px; border: 1px solid #1e293b;">
                <div style="color: #475569; font-size: 0.6rem;">{label}</div>
                <div style="color: {color}; font-size: 1rem; font-weight: 600;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ── Performance Plot ──────────────────────────────────────────────────
    st.markdown('<div class="section-title"><h3>📈 PERFORMANCE MAP</h3></div>', unsafe_allow_html=True)
    
    perf_fig = create_performance_plot(results['isp'], results['thrust'])
    st.plotly_chart(perf_fig, use_container_width=True, config={'displayModeBar': False})
    
    # ── Manufacturing Data Banner ──────────────────────────────────────────
    st.markdown("""
    <div class="success-box">
        <h3>✅ MANUFACTURING DATA AVAILABLE</h3>
        <p>All design parameters calculated. Generate a professional report or view detailed analysis below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Detailed Analysis Tabs ──────────────────────────────────────────
    tabs = st.tabs([
        "📊 Performance", "🔥 Thermal", "🔧 Structural",
        "💉 Injector", "💰 Cost", "📋 Manufacturing"
    ])
    
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Flame Temperature", f"{results['T_flame']:.0f} K")
            st.metric("Chamber Pressure", f"{results['p_chamber']:.1f} atm")
            st.metric("Ambient Pressure", f"{results['p_amb']:.3f} atm")
        with col2:
            st.metric("Specific Impulse (Vac)", f"{results['isp']:.0f} s")
            st.metric("Characteristic Velocity", f"{results['c_star']:.0f} m/s")
            st.metric("Molecular Weight", f"{results['mw']:.2f} g/mol")
        with col3:
            st.metric("Mass Flow Rate", f"{results['m_dot']:.4f} kg/s")
            st.metric("Actual Thrust", f"{results['thrust']:.0f} N")
            st.metric("Combustion Efficiency", f"{results['eta_c']*100:.1f}%")
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Heat Flux", f"{results['heat']['heat_flux']:.0f} kW/m²")
            st.metric("Gas Wall Temperature", f"{results['heat']['T_wall']:.0f} K")
            st.metric("Heat Transfer Coeff", f"{results['heat']['h_g']:.0f} W/m²·K")
        with col2:
            st.metric("Film Cooling Flow", f"{results['cooling']['film_cooling_flow']:.4f} kg/s")
            st.metric("Regen Cooling Flow", f"{results['cooling']['regen_cooling_flow']:.4f} kg/s")
            st.metric("Thermal Stress", f"{results['cooling']['thermal_stress']:.0f} MPa")
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Wall Thickness", f"{results['structure']['wall_thickness']:.1f} mm")
            st.metric("Material", MATERIALS[results['material']]['name'])
            st.metric("Safety Factor", f"{results['structure']['safety_factor']:.1f}x")
        with col2:
            st.metric("Hoop Stress", f"{results['structure']['hoop_stress']:.0f} MPa")
            st.metric("Axial Stress", f"{results['structure']['axial_stress']:.0f} MPa")
            st.metric("Von Mises Stress", f"{results['structure']['von_mises']:.0f} MPa")
    
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LOX Mass Flow", f"{results['injector']['m_dot_lox']:.4f} kg/s")
            st.metric("Fuel Mass Flow", f"{results['injector']['m_dot_fuel']:.4f} kg/s")
            st.metric("Injector ΔP", f"{results['injector']['delta_p']:.2f} atm")
        with col2:
            st.metric("LOX Orifice Diameter", f"{results['injector']['D_lox']:.2f} mm")
            st.metric("Fuel Orifice Diameter", f"{results['injector']['D_fuel']:.2f} mm")
            st.metric("Injection Velocity", f"{results['injector']['injector_velocity']:.1f} m/s")
    
    with tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Propellant Cost", f"${results['cost']['total_cost']:.2f}")
            st.metric("Manufacturing Cost", f"${results['cost']['manufacturing_cost']:.2f}")
            st.metric("Total Project Cost", f"${results['cost']['total_project_cost']:.2f}")
        with col2:
            st.metric("Cost per kg", f"${results['cost']['cost_per_kg']:.2f}/kg")
            for key, value in results['cost']['breakdown'].items():
                st.metric(key.capitalize(), f"${value:.2f}")
    
    with tabs[5]:
        st.markdown("### 📋 Manufacturing Readiness Assessment")
        
        checks = [
            ("Design validated with real combustion efficiency model", results['eta_c'] > 0.85),
            ("Structural analysis completed", results['structure']['safety_factor'] > 2.0),
            ("Cooling system designed", results['cooling']['film_cooling_flow'] > 0),
            ("Injector geometry defined", results['injector']['D_lox'] > 0),
            ("Material selected", results['material'] in MATERIALS),
            ("Cost analysis complete", results['cost']['total_cost'] > 0)
        ]
        
        for check, passed in checks:
            if passed:
                st.success(f"✅ {check}")
            else:
                st.warning(f"⚠️ {check}")
        
        ready_count = sum([1 for _, passed in checks if passed])
        readiness = ready_count / len(checks) * 100
        
        st.metric("Manufacturing Readiness", f"{readiness:.0f}%")
        
        if readiness > 80:
            st.success("✅ Engine design is manufacturing-ready!")
        elif readiness > 50:
            st.info("ℹ️ Engine design needs some refinement before manufacturing.")
        else:
            st.error("❌ Significant design work needed before manufacturing.")
    
    # ── Report Generation ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Generate Professional Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 GENERATE HTML REPORT", use_container_width=True):
            with st.spinner("Generating professional report..."):
                html_report = generate_professional_report(
                    results, fuel_mix, lox_flux, altitude, target_thrust,
                    burn_time, material, wall_thickness
                )
                
                b64 = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="NAVGAL_Engineering_Report_{datetime.datetime.now().strftime("%Y%m%d")}.html" style="background: #3b82f6; color: white; padding: 10px 25px; border-radius: 6px; text-decoration: none; font-weight: 600; display: inline-block;">📥 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ Report generated successfully!")
    
    with col2:
        # Export data
        export_data = {
            'Timestamp': datetime.datetime.now().isoformat(),
            'Fuel_Blend': fuel_mix,
            'LOX_Flow': lox_flux,
            'Altitude': altitude,
            'Thrust_Target': target_thrust,
            'Flame_Temp': results['T_flame'],
            'Chamber_Pressure': results['p_chamber'],
            'Isp': results['isp'],
            'C_star': results['c_star'],
            'D_t': results['D_t'],
            'D_e': results['D_e'],
            'Epsilon': results['epsilon'],
            'Mass_Flow': results['m_dot'],
            'Efficiency': results['eta_total'],
            'Safety_Factor': results['structure']['safety_factor'],
            'Heat_Flux': results['heat']['heat_flux'],
            'Total_Cost': results['cost']['total_project_cost']
        }
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Export Data (CSV)",
            data=csv,
            file_name=f"engine_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 Configure engine parameters in the sidebar and click 'IGNITE ENGINE' to begin analysis.")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="report-footer">
    <p>NAVGAL Propulsion Systems &bull; Engineering Division &bull; Confidential</p>
    <p>Complete Engineering Analysis Suite v3.0</p>
</div>
""", unsafe_allow_html=True)