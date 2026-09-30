"""
Ravish Theta Hub — one page for the option-income setups (?scanner=ravish_theta)
================================================================================
Folds the option-income pages into one, switched by a selector (URL: &view=...):

  * machine  -> Ravish Theta Machine: earnings iron fly, VIX calendar, post-earnings condor
  * calendar_em -> expected-move call / put / double calendars (video GuI-hH_jhlg)
  * spx_put  -> SPX RSI < 35 bull put spread (was ?scanner=spx_rsi_put)
  * wheel    -> Theta Wheel weekly CSPs (was ?scanner=theta_wheel)

Old ?scanner=spx_rsi_put / theta_wheel links are redirected here by dashboard.py.
Each view is rendered lazily, so only the selected one does any work.
"""
from __future__ import annotations

import importlib
import sys

import streamlit as st

VIEWS = {
    "machine": ("⏳ Theta Machine (fly · calendar · condor)", "ravish_theta_module", "render_ravish_theta"),
    "calendar_em": ("📅 EM Calendar (call · put · double)", "calendar_em_module", "render_calendar_em"),
    "spx_put": ("📉 SPX RSI Bull Put",                       "spx_rsi_put_module",  "render_spx_rsi_put"),
    "wheel":   ("🎡 Theta Wheel (weekly CSPs)",               "theta_wheel_module",  "render_theta_wheel"),
}
LEGACY_SCANNER_VIEWS = {"spx_rsi_put": "spx_put", "theta_wheel": "wheel"}


def _sync_url():
    st.query_params["view"] = st.session_state["ravish_view"]
    st.session_state["_ravish_url_view"] = st.session_state["ravish_view"]


def render_ravish_hub():
    # URL wins when it changes (bookmarks, legacy redirects); the radio wins otherwise.
    url_view = st.query_params.get("view")
    if url_view in VIEWS and st.session_state.get("_ravish_url_view") != url_view:
        st.session_state["ravish_view"] = url_view
        st.session_state["_ravish_url_view"] = url_view
    if st.session_state.get("ravish_view") not in VIEWS:
        st.session_state["ravish_view"] = "machine"

    view = st.radio(
        "Setup", list(VIEWS), key="ravish_view", horizontal=True,
        format_func=lambda v: VIEWS[v][0], label_visibility="collapsed", on_change=_sync_url,
    )
    st.divider()
    _, module, attr = VIEWS[view]
    try:
        mod = importlib.import_module(module)
    except KeyError:
        # Streamlit's file watcher can drop a module from sys.modules mid-import
        # during a hot reload -> KeyError('<module>'). A second import succeeds.
        sys.modules.pop(module, None)
        mod = importlib.import_module(module)
    getattr(mod, attr)()
