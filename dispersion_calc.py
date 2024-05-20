import numpy as np
import sympy as sm
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq

gamma = sm.Symbol("gamma")
mu_0 = sm.Symbol("mu_0")
H_0 = sm.Symbol("H_0")
M_0 = sm.Symbol("M_0")
n = sm.Symbol("n")
q = sm.Symbol("q")  # exchange constant
omega_H = sm.Symbol("omega_H")
omega_M = sm.Symbol("omega_M")
theta_k = sm.Symbol("theta_k")
L = sm.Symbol("L")
k = sm.Symbol("k")
k_n = sm.Symbol("k_n")
kappa_n = sm.Symbol("kappa_n")
P_nn = sm.Symbol("P_{nn}")
F_nn = sm.Symbol("F_{nn}")
F_n = sm.Symbol("F_n")

kappa_n = n * sm.pi / L
k_n = sm.sqrt(k**2 + kappa_n**2)
omega_H = gamma * H_0
omega_M = gamma * M_0
F_n = 2 / (sm.Abs(k) * L) * (1 - (-1) ** n * sm.exp(-sm.Abs(k) * L))
P_nn = k**2 / k_n**2 - k**4 / k_n**4 * F_n / (
    1 + sm.KroneckerDelta(n, 0)
)
F_nn = (
    1
    - P_nn * (sm.cos(theta_k)) ** 2
    + omega_M
    * (P_nn * (1 - P_nn) * (sm.sin(theta_k)) ** 2)
    / (omega_H + q * omega_M * k_n**2)
)
omega_n = sm.sqrt(
    (omega_H + q * omega_M * k_n**2)
    * (omega_H + q * omega_M * k_n**2 + omega_M * F_nn)
)


def magnon_dispersion(
    L_val, M_0_val, ex_stiff_val, gamma_val, theta_k_val, H_0_val
):
    gamma_val = 2 * np.pi * gamma_val  # dimension: Hz/T
    mu_0_val = 4 * np.pi * 1e-7  # dimension: H/m
    M_0_val_SI = M_0_val / mu_0_val  # unit: A/m
    q_val = ex_stiff_val * 2 / M_0_val_SI / M_0_val  # dimension: m^2
    omega = (
        omega_n.subs(gamma, gamma_val)
        .subs(M_0, M_0_val)
        .subs(n, 0)
        .subs(q, q_val)
        .subs(L, L_val)
    )
    f_magnon = omega * 1e-9 / (2 * np.pi)
    f_magnon = (
        f_magnon.subs(theta_k, theta_k_val / 180 * np.pi).subs(H_0, H_0_val),
    )[0]
    return sm.lambdify(k, f_magnon)


def plot_dispersion(dispersion_dict, k_min, k_max, x_log, y_log):
    fig = go.Figure()
    for disp_name, disp_func in dispersion_dict.items():
        if x_log is True:
            k_array = np.logspace(np.log10(k_min), np.log10(k_max), 101, base=10)
        else:
            k_array = np.linspace(k_min, k_max, 101)
        f_array = disp_func(k_array)
        fig.add_trace(
            go.Scatter(
                x=k_array,
                y=f_array,
                name=disp_name,
                showlegend=True,
            ),
        )
    if x_log is True:
        fig.update_xaxes(
            type="log",
            exponentformat="power",
        )
    else:
        fig.update_xaxes(
            showexponent = 'all',
            exponentformat="power",
        )
    fig.update_xaxes(
        title_text="k (1/m)",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    if y_log is True:
        fig.update_yaxes(
            type="log",
            exponentformat="power",
        )
    else:
        fig.update_yaxes(
            showexponent = 'all',
            exponentformat="power",
        )
    fig.update_yaxes(
        title_text="Frequency (GHz)",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, monospace", size=24, color="Black"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=80, r=40, t=40, b=40),
    )
    return fig


fig = go.Figure()
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Dispersion Calculator"
app.layout = dbc.Container(
    [
        html.Center(html.H1(children="Dispersion relations of Spin-wave and SAW")),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # html.Center(html.H2(children="Parameters")),
                        dbc.Row(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    children="""SAW velocity (m/s)"""
                                                ),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="saw-velocity",
                                                            precision=4,
                                                            value=3.5e3,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    children="""Film Thickness (nm)"""
                                                ),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="film-thickness",
                                                            precision=2,
                                                            value=20,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"height": "15vh"},
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H3(children="""Ms (T)"""),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="saturated-magnetization",
                                                            precision=3,
                                                            value=0.98,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    children="""Exchange (J/m)"""
                                                ),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="exchange-constant",
                                                            precision=3,
                                                            value=1.05e-11,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    children="""Gamma (Hz/T)"""
                                                ),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="gamma-constant",
                                                            precision=3,
                                                            value=2.98e10,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"height": "20vh"},
                                ),
                                dbc.Row(
                                    [
                                        html.Center(
                                            html.H3(id="magnetic-field-output")
                                        ),
                                        dcc.Slider(
                                            id="magnetic-field",
                                            min=0,
                                            max=100,
                                            value=10,
                                            step=0.2,
                                            marks={
                                                0: {"label": "0"},
                                                10: {"label": "10"},
                                                20: {"label": "20"},
                                                30: {"label": "30"},
                                                40: {"label": "40"},
                                                50: {"label": "50"},
                                                60: {"label": "60"},
                                                70: {"label": "70"},
                                                80: {"label": "80"},
                                                90: {"label": "90"},
                                                100: {"label": "100"},
                                            },
                                        ),
                                    ],
                                    style={"height": "15vh"},
                                ),
                                dbc.Row(
                                    [
                                        html.Center(
                                            html.H3(
                                                id="magnetic-field-angle-output"
                                            )
                                        ),
                                        dcc.Slider(
                                            id="angle",
                                            min=0,
                                            max=90,
                                            value=45,
                                            step=1,
                                            marks={
                                                0: {"label": "0"},
                                                15: {"label": "15"},
                                                30: {"label": "30"},
                                                45: {"label": "45"},
                                                60: {"label": "60"},
                                                75: {"label": "75"},
                                                90: {"label": "90"},
                                            },
                                        ),
                                    ],
                                    style={"height": "15vh"},
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    children="""k min (1/m)"""
                                                ),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="k-min",
                                                            precision=2,
                                                            value=1.0e5,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    children="""k max (1/m)"""
                                                ),
                                                html.Center(
                                                    [
                                                        daq.PrecisionInput(
                                                            id="k-max",
                                                            precision=2,
                                                            value=1.0e7,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"height": "10vh"},
                                ),
                            ],
                            # className="bg-light",
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.Center(
                            [
                                # html.H2(children="Plot"),
                                dbc.Row([
                                    dbc.Col(daq.ToggleSwitch(
                                            id="x-log-scale-switch",
                                            label="x: Linear/Log",
                                            value=False,
                                            # labelPosition="bottom"
                                        )),
                                    dbc.Col(daq.ToggleSwitch(
                                            id="y-log-scale-switch",
                                            label="y: Linear/Log",
                                            value=False,
                                            # labelPosition="bottom"
                                        ))
                                ]),
                            ]
                        ),
                        dbc.Row(
                            [
                                dcc.Graph(id="dispersion-graph", figure=fig),
                            ],
                            style={"height": "70vh"},
                        ),
                    ],
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("dispersion-graph", "figure"),
    [
        Input("magnetic-field", "value"),
        Input("angle", "value"),
        Input("film-thickness", "value"),
        Input("saw-velocity", "value"),
        Input("saturated-magnetization", "value"),
        Input("exchange-constant", "value"),
        Input("gamma-constant", "value"),
        Input("k-min", "value"),
        Input("k-max", "value"),
        Input("x-log-scale-switch", "value"),
        Input("y-log-scale-switch", "value"),
    ],
)
def update_plot(
    field_val,
    angle_val,
    L_val,
    v_saw,
    M_s_val,
    ex_stiff_val,
    gamma_val,
    k_min,
    k_max,
    x_log_enable,
    y_log_enable,
):
    disp_dict = {}
    func_saw = sm.lambdify(
        k, sm.Abs(v_saw * k * sm.cos(angle_val / 180 * sm.pi)) * 1e-9
    )
    disp_dict["SAW"] = func_saw
    disp_dict["BVW"] = magnon_dispersion(
        L_val * 1e-9, M_s_val, ex_stiff_val, gamma_val, 0, field_val * 1e-3
    )
    disp_dict["SW"] = magnon_dispersion(
        L_val * 1e-9, M_s_val, ex_stiff_val, gamma_val, 90, field_val * 1e-3
    )
    disp_dict[f"{angle_val} deg."] = magnon_dispersion(
        L_val * 1e-9,
        M_s_val,
        ex_stiff_val,
        gamma_val,
        float(angle_val),
        field_val * 1e-3,
    )
    fig = plot_dispersion(disp_dict, k_min, k_max, x_log_enable, y_log_enable)
    return fig


@app.callback(
    Output("magnetic-field-output", "children"),
    Input("magnetic-field", "value"),
)
def update_output(value):
    return f"Magnetic Field : {value} mT"


@app.callback(
    Output("magnetic-field-angle-output", "children"), Input("angle", "value")
)
def update_output(value):
    return f"Magnetic Field Angle : {value}°"


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=True)
