import streamlit as st
import lifefit as lf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import base64


@st.cache_data
def load_LifeData():
    fluorstr = requests.get(
        "https://raw.githubusercontent.com/RNA-FRETools/Lifefit/master/data/lifetime/Atto550_DNA.txt"
    )
    irfstr = requests.get("https://raw.githubusercontent.com/RNA-FRETools/Lifefit/master/data/IRF/irf.txt")
    fluor, timestep_ns = lf.tcspc.read_decay(io.StringIO(fluorstr.text))
    irf, _ = lf.tcspc.read_decay(io.StringIO(irfstr.text))
    return fluor, timestep_ns, irf


@st.cache_data
def load_AnisoData(channels):
    fluorstr = {}
    fluor = {}
    for c in channels:
        fluorstr[c] = requests.get(
            "https://raw.githubusercontent.com/RNA-FRETools/Lifefit/master/data/anisotropy/{}.txt".format(c)
        )
        fluor[c], timestep_ns = lf.tcspc.read_decay(io.StringIO(fluorstr[c].text))
    irfstr = requests.get("https://raw.githubusercontent.com/RNA-FRETools/Lifefit/master/data/IRF/irf.txt")
    irf, _ = lf.tcspc.read_decay(io.StringIO(irfstr.text))
    return fluor, timestep_ns, irf


def fit_LifeData(fluor_life, tau0):
    fluor_life.reconvolution_fit(tau0, verbose=False)
    return fluor_life


def fit_AnisoData(aniso, p0, modelfunc, manual_interval):
    aniso.rotation_fit(p0, modelfunc, manual_interval, verbose=False)
    if modelfunc == "local_global_rotation" or modelfunc == "hindered_rotation":
        aniso.aniso_fraction = aniso._fraction_freeStacked(aniso.fit_param[0], aniso.fit_param[2])
    else:
        aniso.aniso_fraction = (None, None)
    return aniso


def get_LifeFitparams(fluor_life, n_decays):
    fit_parameters = pd.DataFrame(fluor_life.fit_param, columns=["tau", "ampl"])
    fit_parameters["tau"] = [
        "{:0.2f} +/- {:0.2f}".format(val, err)
        for val, err in zip(fluor_life.fit_param["tau"], fluor_life.fit_param_std["tau"])
    ]
    fit_parameters["ampl"] = ["{:0.2f}".format(val) for val in fluor_life.fit_param["ampl"]]
    fit_parameters.index = ["tau{:d}".format(i + 1) for i in range(n_decays)]
    fit_parameters = pd.concat(
        [
            fit_parameters,
            pd.DataFrame(
                {"tau": "{:0.2f} +/- {:0.2f}".format(fluor_life.av_lifetime, fluor_life.av_lifetime_std), "ampl": "-"},
                index=["weighted tau"],
            ),
        ]
    )
    fit_parameters.columns = ["lifetime (ns)", "weight"]
    return fit_parameters


def get_AnisoFitparams(aniso, model):
    fit_parameters = pd.DataFrame(aniso.fit_param, columns=["p"])
    fit_parameters["p"] = [
        "{:0.2f} +/- {:0.2f}".format(aniso.fit_param[i], aniso.fit_param_std[i])
        for i, p in enumerate(aniso.param_names[model])
    ]
    fit_parameters.index = aniso.param_names[model]
    fit_parameters.columns = ["correlation times are in ns"]
    return fit_parameters


def to_base64(df):
    csv = df.to_csv(index=False, float_format="%.3f")
    return base64.b64encode(csv.encode()).decode()


def main():
    st.set_page_config(
        page_title="LifeFit",
        page_icon=":chart_with_downwards_trend:",
    )
    st.title("Welcome to LifeFit")
    st.markdown("## What is LifeFit?")
    st.markdown(
        "LifeFit analyzes time-correlated single-photon counting (TCSPC) data sets, "
        "namely fluorescence lifetime and time-resolved anisotropy decays. "
        "Choose from the sidebar on the left whether you want to look at the demo or analyze your own dataset."
    )

    st.sidebar.image(
        "https://raw.githubusercontent.com/RNA-FRETools/Lifefit/master/docs/images/lifefit_logo.png", width=250
    )

    datatype = st.sidebar.selectbox(
        label="Type of dataset", options=("Fluorescence Lifetime", "Fluorescence Anisotropy")
    )

    mode = st.sidebar.radio(label="Mode", options=("LifeFit Demo", "Analyze your own data"), index=0)
    fileformat = st.sidebar.selectbox("Fileformat", ("Horiba", "time_intensity"))

    st.sidebar.markdown("**Plot settings**")
    show_residuals = st.sidebar.checkbox("Show residuals", True)

    if datatype == "Fluorescence Lifetime":
        lifetime(mode, fileformat, show_residuals)
    else:
        anisotropy(mode, fileformat, show_residuals)


def lifetime(mode, fileformat, show_residuals):

    if mode == "LifeFit Demo":
        st.info("You chose to work with the demo dataset")
        fluor, timestep_ns, irf = load_LifeData()
        fluor_buffer = False
        irf_buffer = False
        gauss_sigma = False
    else:
        st.info("&rarr; Please select your TCSPC dataset")
        fluor_buffer = st.file_uploader("Fluorescence lifetime decay", "txt")
        if fluor_buffer is not None:
            fluor, timestep_ns = lf.tcspc.read_decay(io.TextIOWrapper(fluor_buffer), fileformat)
            if fluor is not None:
                irf_type = st.radio(label="IRF", options=("Gaussian IRF", "experimental IRF"), index=0)
                if irf_type == "experimental IRF":
                    irf_buffer = st.file_uploader("IRF decay", "txt")
                    if irf_buffer is not None:
                        irf, _ = lf.tcspc.read_decay(io.TextIOWrapper(irf_buffer), fileformat)
                        if irf is not None:
                            gauss_sigma = None
                        else:
                            irf = False
                else:
                    irf = None
                    irf_buffer = False
                    gauss_sigma = st.number_input("IRF sigma", min_value=0.00, value=0.10, step=0.01, format="%0.2f")

    if (fluor_buffer is not None) and (irf_buffer is not None):
        if (fluor is not None) and (irf is not False):

            st.markdown("### Start parameters for reconvolution fit")
            n_decays = st.number_input("number of exponential decays", value=2, min_value=1, max_value=4, step=1)
            st.write("---")
            tau0 = []
            col = st.columns(n_decays)
            for i in range(n_decays):
                tau0.append(
                    col[i].number_input(
                        "tau{:d}".format(i + 1),
                        value=float(10**i),
                        step=float(10 ** (i - 1)),
                        format="%0.{prec}f".format(prec=max(1 - i, 0)),
                    )
                )

            fluor_life = lf.tcspc.Lifetime(fluor, timestep_ns, irf, gauss_sigma=gauss_sigma)
            with st.spinner("Fitting..."):
                fluor_life = fit_LifeData(fluor_life, tau0)
                try:
                    fluor_life = fit_LifeData(fluor_life, tau0)
                except:
                    st.warning("Fit did not converge.")
                else:
                    st.success("Reconvolution fit successful!")

                    fit_parameters = get_LifeFitparams(fluor_life, n_decays)
                    st.table(fit_parameters)

                    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1)
                    xlimits = st.slider(
                        "Select a time limits (ns) on the x-axis (does not affect the fit)",
                        min(fluor_life.fluor[fluor_life.fluor[:, 2] > 0, 0]),
                        max(fluor_life.fluor[fluor_life.fluor[:, 2] > 0, 0]),
                        (20.0, 80.0),
                        format="%0.1f ns",
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fluor_life.irf[:, 0],
                            y=fluor_life.irf[:, 2],
                            line=dict(color="rgb(200, 200, 200)", width=1),
                            name="IRF",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fluor_life.fluor[:, 0],
                            y=fluor_life.fluor[:, 2],
                            line=dict(color="rgb(79, 115, 143)", width=1),
                            name="data",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fluor_life.fluor[:, 0], y=fluor_life.fit_y, line=dict(color="black", width=1), name="fit"
                        ),
                        row=1,
                        col=1,
                    )
                    if show_residuals:
                        fig.add_trace(
                            go.Scatter(
                                x=fluor_life.fluor[:, 0],
                                y=fluor_life.fluor[:, 2] - fluor_life.fit_y,
                                line=dict(color="rgb(79, 115, 143)", width=1),
                                name="fit",
                            ),
                            row=2,
                            col=1,
                        )
                        fig.update_layout(
                            yaxis_type="log",
                            template="none",
                            xaxis2_title="time (ns)",
                            yaxis_title="counts",
                            yaxis2_title="residuals",
                            xaxis1=dict(range=xlimits),
                            xaxis2=dict(range=xlimits),
                            yaxis1=dict(range=(0, 4)),
                            showlegend=False,
                        )
                    else:
                        fig.update_layout(
                            yaxis_type="log",
                            template="none",
                            xaxis_title="time (ns)",
                            yaxis_title="counts",
                            yaxis2_title="residuals",
                            xaxis1=dict(range=xlimits),
                            xaxis2=dict(range=xlimits),
                            yaxis1=dict(range=(0, 4)),
                            showlegend=False,
                        )
                    st.plotly_chart(fig, use_container_width=True)

                    data, parameters = fluor_life._serialize()
                    data_df = pd.DataFrame(data)
                    data_df = data_df.loc[(data_df.time >= xlimits[0]) & (data_df.time <= xlimits[1])]
                    st.write("### Export the data and fit parameters")
                    if st.checkbox("Show TCSPC data (json)"):
                        st.write(
                            "**Note:** The entire json formatted TCSPC dataset can be copied to the clipboard by clicking on the topmost blue chart icon."
                        )
                        st.json(data)
                    if st.checkbox("Show TCSPC data (table)"):
                        b64 = to_base64(data_df)
                        href = f'<a href="data:file/csv;base64,{b64}" download="lifetime.csv">Download as .csv</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.table(data_df)
                    if st.checkbox("Show fit parameters (json)"):
                        st.write(
                            "**Note:** The json formatted fit parameters can be copied to the clipboard by clicking on the topmost blue chart icon."
                        )
                        st.json(parameters)
        else:
            st.error("File has a wrong format.")


def anisotropy(mode, fileformat, show_residuals):
    channels = ["VV", "VH", "HV", "HH"]
    fluor_buffer = {}
    fluor = {}
    if mode == "LifeFit Demo":
        st.info("You chose to work with the demo dataset")
        fluor, timestep_ns, irf = load_AnisoData(channels)
        irf_buffer = False
        gauss_sigma = False
    else:
        st.info("&rarr; Please select your TCSPC dataset")
        for c in channels:
            fluor_buffer[c] = st.file_uploader("{} decay".format(c), "txt")
        if all([fb is not None for fb in fluor_buffer.values()]):
            for c in channels:
                fluor[c], timestep_ns = lf.tcspc.read_decay(io.TextIOWrapper(fluor_buffer[c]), fileformat)
            if all([f is not None for f in fluor.values()]):
                irf_type = st.radio(label="IRF", options=("Gaussian IRF", "experimental IRF"), index=0)
                if irf_type == "experimental IRF":
                    irf_buffer = st.file_uploader("IRF decay", "txt")
                    if irf_buffer is not None:
                        irf, _ = lf.tcspc.read_decay(io.TextIOWrapper(irf_buffer))
                        if irf is not None:
                            gauss_sigma = None
                        else:
                            irf = False
                else:
                    irf = None
                    irf_buffer = False
                    gauss_sigma = st.number_input("IRF sigma", min_value=0.00, value=0.10, step=0.01, format="%0.2f")

    if all([fb is not None for fb in fluor_buffer.values()]) and (irf_buffer is not None):
        if all([f is not None for f in fluor.values()]) and (irf is not False):

            fluor_life = {}
            for c in channels:
                fluor_life[c] = lf.tcspc.Lifetime(fluor[c], timestep_ns, irf, gauss_sigma=gauss_sigma)

            aniso = lf.tcspc.Anisotropy(fluor_life["VV"], fluor_life["VH"], fluor_life["HV"], fluor_life["HH"])

            st.markdown("### Start parameters for Anisotropy fit")
            model = st.selectbox(
                "Anisotropy fit model",
                options=("one rotation", "two rotations", "hindered rotation", "local-global rotation"),
            )
            if st.checkbox("Manual interval", False):
                manual_interval = []
                default_start_idx = np.argmax(fluor_life["VV"].fluor[:, 2])
                default_start = np.round(fluor_life["VV"].fluor[default_start_idx, 0], 1)
                channel_stop = np.round(
                    default_start
                    + fluor_life["VV"].fluor[
                        np.argmax(
                            fluor_life["VV"].fluor[default_start_idx:, 2] < 0.01 * max(fluor_life["VV"].fluor[:, 2])
                        ),
                        0,
                    ],
                    1,
                )
                manual_interval.append(
                    st.number_input("start", value=default_start, step=0.1, format="%0.1f", min_value=0.0)
                )
                manual_interval.append(
                    st.number_input("stop", value=channel_stop, step=0.1, format="%0.1f", min_value=0.0)
                )
            else:
                manual_interval = None

            p0 = []
            if model == "one rotation":
                col1, col2 = st.columns(2)
                p0.append(col1.number_input("r0", value=0.4, step=0.1, format="%0.1f", min_value=0.0, max_value=0.4))
                p0.append(col2.number_input("tau_r", value=1.0, step=0.1, format="%0.1f", min_value=0.0))
                modelfunc = "one_rotation"
            if model == "two rotations":
                col1, col2, col3, col4 = st.columns(4)
                p0.append(col1.number_input("r0", value=0.4, step=0.1, format="%0.1f", min_value=0.0, max_value=0.4))
                p0.append(col2.number_input("b", value=0.5, step=0.1, format="%0.1f", min_value=0.0, max_value=1.0))
                p0.append(col3.number_input("tau_r1", value=1.0, step=0.1, format="%0.1f", min_value=0.0))
                p0.append(col4.number_input("tau_r2", value=10.0, step=1.0, format="%0.0f", min_value=0.0))
                modelfunc = "two_rotations"
            if model == "hindered rotation":
                col1, col2, col3 = st.columns(3)
                p0.append(col1.number_input("r0", value=0.4, step=0.1, format="%0.1f", min_value=0.0, max_value=0.4))
                p0.append(col2.number_input("tau_r", value=1.0, step=0.1, format="%0.1f", min_value=0.0))
                p0.append(col3.number_input("r_inf", value=0.1, step=0.1, format="%0.1f", min_value=0.0, max_value=0.4))
                modelfunc = "hindered_rotation"
            if model == "local-global rotation":
                col1, col2, col3, col4 = st.columns(4)
                p0.append(col1.number_input("r0", value=0.4, step=0.1, format="%0.1f", min_value=0.0, max_value=0.4))
                p0.append(col2.number_input("tau_rloc", value=1.0, step=0.1, format="%0.1f", min_value=0.0))
                p0.append(col3.number_input("r_inf", value=0.1, step=0.1, format="%0.1f", min_value=0.0, max_value=0.4))
                p0.append(col4.number_input("tau_rglob", value=10.0, step=1.0, format="%0.0f", min_value=0.0))
                modelfunc = "local_global_rotation"

            with st.spinner("Fitting..."):
                try:
                    aniso = fit_AnisoData(aniso, p0, modelfunc, manual_interval)
                except:
                    st.warning("Fit did not converge.")
                else:
                    st.success("Anisotropy fit successful!")

                    fit_parameters = get_AnisoFitparams(aniso, modelfunc)
                    st.table(fit_parameters)
                    if modelfunc == "local_global_rotation" or modelfunc == "hindered_rotation":
                        st.write(
                            "Dye populations: **free**: {:0.0f}%    **stacked**: {:0.0f}%".format(
                                aniso.aniso_fraction[0] * 100, aniso.aniso_fraction[1] * 100
                            )
                        )

                    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1)
                    xlimits = st.slider(
                        "Select time limits (ns) on the x-axis (does not affect the fit)",
                        min_value=float(min(aniso.time)),
                        max_value=float(max(aniso.time)),
                        value=(float(min(aniso.time)), float(max(aniso.time))),
                        format="%0.1f ns",
                    )
                    fig.add_trace(
                        go.Scatter(x=aniso.time, y=aniso.r, line=dict(color="rgb(79, 115, 143)", width=1), name="data"),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=aniso.time, y=aniso.fit_r, line=dict(color="black", width=1), name="fit"),
                        row=1,
                        col=1,
                    )
                    if show_residuals:
                        fig.add_trace(
                            go.Scatter(
                                x=aniso.time,
                                y=aniso.r - aniso.fit_r,
                                line=dict(color="rgb(79, 115, 143)", width=1),
                                name="fit",
                            ),
                            row=2,
                            col=1,
                        )
                        fig.update_layout(
                            template="none",
                            xaxis2_title="time (ns)",
                            yaxis_title="anisotropy",
                            yaxis2_title="residuals",
                            xaxis1=dict(range=xlimits),
                            xaxis2=dict(range=xlimits),
                            yaxis1=dict(range=(0, 0.4)),
                            showlegend=False,
                        )
                    else:
                        fig.update_layout(
                            template="none",
                            xaxis_title="time (ns)",
                            yaxis_title="anisotropy",
                            yaxis2_title="residuals",
                            xaxis1=dict(range=xlimits),
                            xaxis2=dict(range=xlimits),
                            yaxis1=dict(range=(0, 0.4)),
                            showlegend=False,
                        )
                    st.plotly_chart(fig, use_container_width=True)

                    data, parameters = aniso._serialize()
                    data_df = pd.DataFrame(data)
                    data_df = data_df.loc[(data_df.time >= xlimits[0]) & (data_df.time <= xlimits[1])]
                    st.write("### Export the data and fit parameters")
                    if st.checkbox("Show TCSPC data (json)"):
                        st.write(
                            "**Note:** The entire json formatted TCSPC dataset can be copied to the clipboard by clicking on the topmost blue chart icon."
                        )
                        st.json(data)
                    if st.checkbox("Show TCSPC data (table)"):
                        b64 = to_base64(data_df)
                        href = f'<a href="data:file/csv;base64,{b64}" download="anisotropy.csv">Download as .csv</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.table(data_df)
                    if st.checkbox("Show fit parameters (json)"):
                        st.write(
                            "**Note:** The json formatted fit parameters can be copied to the clipboard by clicking on the topmost blue chart icon."
                        )
                        st.json(parameters)
        else:
            st.error("Files have a wrong format.")


if __name__ == "__main__":
    main()
