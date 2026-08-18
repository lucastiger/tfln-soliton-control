"""Machine-checkable map from paper equations to the code that implements them.

Every entry in :data:`CHANNEL_EQUATIONS` is keyed by a *switch* field name of
:class:`simulator.noise_config.NoiseConfig`, so ``tests/test_equation_map.py``
can enumerate the dataclass and fail the build the moment a channel is added
without a corresponding equation record. The map is the single place that
answers, for a reviewer holding the paper: *which line of code is this
equation, and which test pins it?*

Sourcing policy
---------------
Every field is transcribed from the working tree or from the repository's own
audit (``docs/NOISE_CHANNEL_INVENTORY.md``); nothing here is inferred from
outside knowledge of the reference. Where the repository does **not** record a
paper equation or section number for a channel, the field is the EMPTY STRING
rather than a plausible-looking guess -- a wrong cross-reference in a paper
table is worse than a visible gap. The gaps as committed are:

* ``trn`` -- ``Eqs. 129-130`` are recorded (the spectral models), but no
  section number appears anywhere in the tree.
* ``pyro_eo``, ``tccr`` -- no equation or section recorded. Both are
  platform-specific (chi-2) channels; ``noise_models.py`` cites an unnamed
  "TCCR paper (DOI to be added)" for the pyro-EO sign convention.
* ``thermal_feedback`` -- deterministic thermo-optic dynamics, not a noise
  equation of the reference.

Reference: Herr, Tikan & Kippenberg, arXiv:2604.05897v1 (7 Apr 2026). The
version pin matters: equation and section numbers are v1 numbers.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "ADDITIVITY",
    "CHANNEL_EQUATIONS",
    "ChannelEquation",
    "ENTERS_AS",
    "render_latex",
    "render_markdown",
]

#: Allowed values of :attr:`ChannelEquation.enters_as` -- the four distinct
#: places a channel can touch the equation of motion. Pinned as a closed
#: vocabulary so the table stays comparable across rows.
ENTERS_AS: tuple[str, ...] = (
    "additive field increment",
    "detuning phase rotation",
    "mode-linear detuning",
    "drive amplitude scale",
)

#: Allowed values of :attr:`ChannelEquation.additive_or_multiplicative`.
ADDITIVITY: tuple[str, ...] = ("additive", "multiplicative", "mixed")


class ChannelEquation(NamedTuple):
    """One noise/dynamics channel, from paper equation to pinning test.

    Parameters
    ----------
    channel : str
        Matches a :class:`~simulator.noise_config.NoiseConfig` boolean field
        EXACTLY, which is what lets the test suite enumerate the dataclass and
        detect a channel added without a record here.
    paper_eq : str
        Equation label, e.g. ``"Eq. 126"``. Empty string means "not recorded in
        this repository" -- never a guess.
    paper_section : str
        Section label, e.g. ``"Sec. V.B.2"``. Empty string as above.
    arxiv_id : str
        arXiv identifier of the reference, normally ``"2604.05897"``.
    arxiv_version : str
        Version pin, normally ``"v1"``. Mandatory: equation and section numbers
        are version-specific.
    equation_latex : str
        The CONTINUUM equation, as the reference states it.
    discrete_latex : str
        The DISCRETIZED form actually implemented -- what the code computes.
    code_symbol : str
        Dotted path importable via :mod:`importlib` (module plus attribute
        chain).
    code_file : str
        Repo-relative path of the file holding ``code_symbol``.
    enters_as : str
        One of :data:`ENTERS_AS`: where in the equation of motion the channel
        acts.
    additive_or_multiplicative : str
        One of :data:`ADDITIVITY`, as seen by the intracavity field E.
    colour : str
        Spectrum of the injected quantity, e.g. ``"white"`` or
        ``"lorentzian(tau_th)"``.
    shares_random_source_with : tuple of str
        Other channels drawing from the SAME realization. Must be symmetric --
        if A names B, B must name A.
    units : str
        Units of the quantity actually injected, e.g. ``"rad/s"`` for a
        detuning noise or ``"sqrt(J)"`` for a field increment.
    validated_by : tuple of str
        Names of the test functions in ``tests/`` that pin this channel.

    Raises
    ------
    TypeError
        From ``NamedTuple`` construction if any field is missing. The
        module-local ``_eq`` helper fills the two arXiv fields, so table
        entries only state what is channel-specific.

    Notes
    -----
    A ``NamedTuple`` rather than a dataclass so that entries are immutable,
    hashable and cheap to iterate over in table order, and so the record can be
    unpacked positionally in a renderer.

    Both LaTeX fields are carried because the distinction is the point of the
    map: a reviewer checking an implementation needs the equation the paper
    states AND the discretization that was actually integrated, since almost
    every discrepancy between two codes lives in the gap between them.

    Examples
    --------
    >>> entry = CHANNEL_EQUATIONS["quantum_vacuum"]
    >>> entry.paper_eq, entry.paper_section
    ('Eq. 126', 'Sec. V.B.2')
    >>> entry.enters_as, entry.colour
    ('additive field increment', 'white')
    >>> entry.code_symbol
    'simulator.lle_solver._qnoise_increment'

    A gap is recorded as an empty string, never as a plausible guess:

    >>> CHANNEL_EQUATIONS["trn"].paper_eq, CHANNEL_EQUATIONS["trn"].paper_section
    ('Eqs. 129-130', '')

    Shared random sources are stated symmetrically:

    >>> CHANNEL_EQUATIONS["pyro_eo"].shares_random_source_with
    ('trn', 'fsr')
    """

    channel: str
    """Matches a :class:`~simulator.noise_config.NoiseConfig` boolean field exactly."""

    paper_eq: str
    """e.g. ``"Eq. 126"``. Empty string = not recorded in this repository."""

    paper_section: str
    """e.g. ``"Sec. V.B.2"``. Empty string = not recorded in this repository."""

    arxiv_id: str
    arxiv_version: str

    equation_latex: str
    """The continuum equation, as the reference states it."""

    discrete_latex: str
    """The discretized form actually implemented (what the code computes)."""

    code_symbol: str
    """Dotted path, importable via :mod:`importlib` (module + attribute chain)."""

    code_file: str
    """Repo-relative path of the file holding ``code_symbol``."""

    enters_as: str
    """One of :data:`ENTERS_AS`."""

    additive_or_multiplicative: str
    """One of :data:`ADDITIVITY`, as seen by the intracavity field E."""

    colour: str
    """Spectrum of the injected quantity, e.g. ``"white"``, ``"lorentzian(tau_th)"``."""

    shares_random_source_with: tuple[str, ...]
    """Other channels drawing from the SAME realization. Must be symmetric."""

    units: str
    """Units of the quantity actually injected."""

    validated_by: tuple[str, ...]
    """Test function names (in ``tests/``) that pin this channel."""


_ARXIV_ID = "2604.05897"
_ARXIV_VERSION = "v1"


def _eq(**kwargs) -> ChannelEquation:
    kwargs.setdefault("arxiv_id", _ARXIV_ID)
    kwargs.setdefault("arxiv_version", _ARXIV_VERSION)
    return ChannelEquation(**kwargs)


CHANNEL_EQUATIONS: dict[str, ChannelEquation] = {
    # ------------------------------------------------------------------
    "quantum_vacuum": _eq(
        channel="quantum_vacuum",
        paper_eq="Eq. 126",
        paper_section="Sec. V.B.2",
        equation_latex=(
            r"\partial_t E = \dots + \sqrt{\kappa}\,\hat\xi_\mu(t),\quad "
            r"\langle \hat\xi_\mu(t)\,\hat\xi^{\dagger}_{\mu'}(t')\rangle "
            r"= \delta(t-t')\,\delta_{\mu\mu'}"
        ),
        # Truncated-Wigner c-number limit: <xi xi*> = 1/2 delta delta, injected
        # in the TIME domain once per fine step (lle_solver.py::_qnoise_increment).
        discrete_latex=(
            r"E_j \leftarrow E_j + \sigma_q\,(g_j + i\,h_j),\quad "
            r"\sigma_q = \sqrt{\hbar\omega_0\,\kappa\,n_\tau\,\Delta t/4},\quad "
            r"g_j,h_j \sim \mathcal{N}(0,1)\ \text{i.i.d. per fast-time sample},\ "
            r"\Delta t = t_R/M\ \text{(or } t_R \text{ at roundtrip cadence)}"
        ),
        code_symbol="simulator.lle_solver._qnoise_increment",
        code_file="simulator/lle_solver.py",
        enters_as="additive field increment",
        additive_or_multiplicative="additive",
        colour="white",
        shares_random_source_with=(),
        units="sqrt(J) (field increment; |E|^2 in J)",
        validated_by=(
            "test_injection_variance_matches_prescription",
            "test_vacuum_equilibrium_occupation",
            "test_vacuum_seed_occupancy_half_photon",
            "test_disabled_path_adds_no_rng_to_scan_body",
            "test_config_missing_keys_default_off",
        ),
    ),
    # ------------------------------------------------------------------
    "trn": _eq(
        channel="trn",
        paper_eq="Eqs. 129-130",
        # No section number for the thermorefractive channel appears anywhere
        # in the tree -- left empty rather than guessed.
        paper_section="",
        equation_latex=(
            r"\delta\omega_{\mathrm{TRN}}(t) = C_{\mathrm{pull}}\,\delta T(t),\quad "
            r"C_{\mathrm{pull}} = \frac{\omega_0}{n_0}\left(\frac{dn}{dT} "
            r"+ n_0\alpha_L\right),\quad "
            r"\langle \delta T^2\rangle = \frac{k_B T^2}{\rho C_p V},\quad "
            r"S_{\delta T}(f) = \frac{4\,\langle\delta T^2\rangle\,\tau_{\mathrm{th}}}"
            r"{1 + (2\pi f \tau_{\mathrm{th}})^2}"
        ),
        # single_pole = the historical AR(1); colored models synthesize from
        # the PSD host-side (simulator/colored_noise.py::synthesize_from_psd).
        discrete_latex=(
            r"\delta T_{k+1} = a\,\delta T_k + \sigma\sqrt{1-a^2}\;w_k,\quad "
            r"a = e^{-t_R/\tau_{\mathrm{th}}},\ w_k \sim \mathcal{N}(0,1);\quad "
            r"\delta\omega_k = C_{\mathrm{pull}}\,\delta T_k"
        ),
        code_symbol="simulator.noise_models.TRNoise",
        code_file="simulator/noise_models.py",
        enters_as="detuning phase rotation",
        additive_or_multiplicative="multiplicative",
        colour="lorentzian(tau_th) | kondratiev_gorodetsky | csv",
        shares_random_source_with=("pyro_eo", "fsr"),
        units="rad/s (detuning)",
        validated_by=(
            "test_single_pole_bit_identical_to_legacy_ar1",
            "test_trn_disabled_returns_exact_zeros",
            "test_tk_zero_collapses_all_delta_t_channels",
            "test_variance_conserved_across_psd_models",
        ),
    ),
    # ------------------------------------------------------------------
    "pyro_eo": _eq(
        channel="pyro_eo",
        paper_eq="",
        paper_section="",
        equation_latex=(
            r"\delta\omega_{\mathrm{pyro}}(t) = C_{\mathrm{pyro}}\,\delta T(t),\quad "
            r"C_{\mathrm{pyro}} = \frac{\omega_0\,n_0^2\,r_{33}\,p}"
            r"{2\,\varepsilon_0\,\varepsilon_{r,\mathrm{eff}}}"
        ),
        # SAME delta_T array as TRN; combined with a MINUS sign in TotalNoise.
        discrete_latex=(
            r"\delta\omega^{\mathrm{tot}}_k = C_{\mathrm{pull}}\,\delta T_k "
            r"- C_{\mathrm{pyro}}\,\delta T_k + \delta\omega^{\mathrm{TCCR}}_k,\quad "
            r"\delta T_k\ \text{the identical realization used by TRN}"
        ),
        code_symbol="simulator.noise_models.PyroEONoise",
        code_file="simulator/noise_models.py",
        enters_as="detuning phase rotation",
        additive_or_multiplicative="multiplicative",
        colour="lorentzian(tau_th) | kondratiev_gorodetsky | csv",
        shares_random_source_with=("trn", "fsr"),
        units="rad/s (detuning)",
        validated_by=(
            "test_sample_with_delta_t_consistency",
            "test_csv_delta_omega_units_share_delta_t_with_pyroeo",
            "test_tk_zero_collapses_all_delta_t_channels",
        ),
    ),
    # ------------------------------------------------------------------
    "tccr": _eq(
        channel="tccr",
        paper_eq="",
        paper_section="",
        equation_latex=(
            r"\frac{d\omega}{dN_s} = -\frac{\omega_0 n_0^2 r_{33}}{2}\,"
            r"\frac{e}{\varepsilon_0 \varepsilon_{r} A_{\mathrm{eff}}},\quad "
            r"S_{\mathrm{TCCR}}(f) = \frac{S_0}{1+(2\pi f\tau_c)^2},\quad "
            r"S_0 = \left(\frac{d\omega}{dN_s}\right)^2 N_{s,\mathrm{eq}}\,2\tau_c"
        ),
        discrete_latex=(
            r"\delta\omega^{\mathrm{TCCR}}_{k+1} = a_c\,\delta\omega^{\mathrm{TCCR}}_k "
            r"+ \sigma_{\mathrm{TCCR}}\sqrt{1-a_c^2}\;w_k,\quad "
            r"a_c = e^{-t_R/\tau_c},\ \tau_c = \tau_{\mathrm{carrier}}"
        ),
        code_symbol="simulator.noise_models.TCCRNoise",
        code_file="simulator/noise_models.py",
        enters_as="detuning phase rotation",
        additive_or_multiplicative="multiplicative",
        colour="lorentzian(tau_carrier)",
        # Independent stream: second subkey of the unconditional split in
        # TotalNoise.sample_with_delta_t.
        shares_random_source_with=(),
        units="rad/s (detuning)",
        validated_by=(
            "test_disabling_trn_does_not_shift_tccr_stream",
            "test_single_pole_bit_identical_to_legacy_ar1",
        ),
    ),
    # ------------------------------------------------------------------
    "pump_freq_noise": _eq(
        channel="pump_freq_noise",
        paper_eq="",
        paper_section="Sec. V.B.4",
        equation_latex=(
            r"S_{\delta\nu}(f) = h_0 + \frac{h_{-1}}{f}\ [\mathrm{Hz^2/Hz}],\quad "
            r"\Delta\nu_L = \pi h_0,\quad "
            r"\delta\omega \equiv \omega_{\mathrm{res}} - \omega_p "
            r"\Rightarrow \delta\omega \mathrel{+}= -2\pi\,\delta\nu_p(t)"
        ),
        discrete_latex=(
            r"\delta\nu_k = w_k\sqrt{h_0 f_s/2} + \mathrm{FFT}^{-1}"
            r"\!\left[\sqrt{h_{-1}/\max(f,f_1)}\right]_k,\quad f_s = 1/t_R,\ "
            r"f_1 = f_s/N;\quad \delta\omega_k \mathrel{+}= -2\pi\,\delta\nu_k"
        ),
        code_symbol="simulator.noise_models.PumpNoise.sample_freq",
        code_file="simulator/noise_models.py",
        enters_as="detuning phase rotation",
        additive_or_multiplicative="multiplicative",
        colour="h0 + h-1/f",
        shares_random_source_with=(),
        units="rad/s (detuning; sample_freq returns 2*pi*delta_nu)",
        validated_by=(
            "test_freq_noise_psd_fidelity",
            "test_sign_convention_exact",
            "test_linear_response_transfer",
            "test_flag_off_bit_identical_to_legacy",
        ),
    ),
    # ------------------------------------------------------------------
    "pump_rin": _eq(
        channel="pump_rin",
        paper_eq="",
        paper_section="Sec. V.B.5",
        equation_latex=(
            r"P_{\mathrm{in}}(t) = \bar P_{\mathrm{in}}\,(1+\epsilon(t)),\quad "
            r"S_\epsilon(f) = 10^{\mathrm{floor}/10} "
            r"+ 10^{\mathrm{excess}/10}\frac{f_c}{f}\ (f<f_c),\quad "
            r"S_\epsilon(f) = 10^{\mathrm{floor}/10}\ (f \ge f_c)"
        ),
        discrete_latex=(
            r"s_k = \max(1+\epsilon_k,\,0),\quad "
            r"F_k = \sqrt{\max(\kappa_c \bar P_{\mathrm{in}} s_k,\,0)}\;\Delta t_{\mathrm{sub}},"
            r"\quad E \leftarrow E + F_k\ \text{(pump kick, held over the round trip)}"
        ),
        code_symbol="simulator.noise_models.PumpNoise.sample_rin",
        code_file="simulator/noise_models.py",
        enters_as="drive amplitude scale",
        # Multiplicative on the drive F; F itself enters E additively, and RIN
        # reaches E a second time through P_abs -> DeltaT -> detuning.
        additive_or_multiplicative="mixed",
        colour="rin_floor + rin_excess*(f_c/f) below f_c",
        shares_random_source_with=(),
        units="dimensionless (relative intensity epsilon)",
        validated_by=(
            "test_rin_psd_fidelity",
            "test_rin_energy_balance",
            "test_rin_clip_enforced_and_warns",
            "test_flag_off_bit_identical_to_legacy",
        ),
    ),
    # ------------------------------------------------------------------
    "fsr": _eq(
        channel="fsr",
        paper_eq="",
        paper_section="Sec. V.B.1",
        equation_latex=(
            r"\delta D_1(t) = \frac{D_1}{\omega_0}\,C_{\mathrm{pull}}\,\delta T(t),\quad "
            r"D_1 = 2\pi\,\mathrm{FSR};\quad "
            r"\text{mode } \mu \text{ acquires the detuning } \mu\,\delta D_1(t)"
        ),
        discrete_latex=(
            r"\mathcal{L}_\mu \leftarrow \mathcal{L}_\mu "
            r"- i\,\mu\,\delta D_1(t_k)\,\Delta t_{\mathrm{sub}},\quad "
            r"\delta D_1(t_k) = \frac{D_1}{\omega_0} C_{\mathrm{pull}}\,\delta T_k,\ "
            r"\delta T_k\ \text{regenerated from the SAME noise keys as TRN}"
        ),
        code_symbol="simulator.lle_solver._delta_t_sequences",
        code_file="simulator/lle_solver.py",
        enters_as="mode-linear detuning",
        additive_or_multiplicative="multiplicative",
        colour="lorentzian(tau_th) | kondratiev_gorodetsky | csv",
        shares_random_source_with=("trn", "pyro_eo"),
        units="rad/s (delta_D1; mode mu sees mu*delta_D1)",
        validated_by=(
            "test_fsr_constant_dd1_exact_phase",
            "test_fsr_tk_zero_channel_identically_zero",
            "test_fsr_without_trn_warns_and_is_zero",
        ),
    ),
    # ------------------------------------------------------------------
    "thermal_feedback": _eq(
        channel="thermal_feedback",
        # NOT a noise channel and NOT an equation of the reference: the
        # deterministic thermo-optic ODE. Listed because NoiseConfig carries
        # the switch and a reproducible run must record it.
        paper_eq="",
        paper_section="",
        equation_latex=(
            r"\frac{d\,\Delta T}{dt} = -\frac{\Delta T}{\tau_{\mathrm{th}}} "
            r"+ \frac{\Gamma_{\mathrm{th}}\,P_{\mathrm{abs}}}{\rho C_p V},\quad "
            r"P_{\mathrm{abs}} = \kappa_i U_{\mathrm{int}}/t_R;\quad "
            r"\delta\omega_{\mathrm{th}} = -\frac{\omega_0}{n_0}\frac{dn}{dT}\,\Delta T"
        ),
        discrete_latex=(
            r"\Delta T_{k+1} = \Delta T_k + \Delta t_{\mathrm{fine}}"
            r"\left(-\frac{\Delta T_k}{\tau_{\mathrm{th}}} "
            r"+ \frac{\Gamma_{\mathrm{th}} P_{\mathrm{abs},k}}{\rho C_p V}\right)"
            r"\ \text{(forward Euler, once per fine step)}"
        ),
        code_symbol="simulator.lle_solver._thermal_params",
        code_file="simulator/lle_solver.py",
        enters_as="detuning phase rotation",
        additive_or_multiplicative="multiplicative",
        colour="deterministic (no stochastic source)",
        shares_random_source_with=(),
        units="K (Delta T); enters as rad/s after the thermo-optic pull",
        validated_by=(
            "test_v3_thermal_sign",
            "test_all_off_leaves_thermal_feedback_at_its_passed_value",
        ),
    ),
}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
#: (attribute, column header) in table order.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("channel", "Channel"),
    ("paper_eq", "Paper eq."),
    ("paper_section", "Section"),
    ("enters_as", "Enters as"),
    ("additive_or_multiplicative", "Add/mult"),
    ("colour", "Colour"),
    ("units", "Units"),
    ("code_symbol", "Code symbol"),
    ("shares_random_source_with", "Shares source with"),
    ("validated_by", "Validated by"),
)

_EMPTY = "--"


def _cell(value: object) -> str:
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value) if value else _EMPTY
    text = str(value)
    return text if text else _EMPTY


def _md_escape(text: str) -> str:
    """Escape the characters that would break a Markdown table cell."""
    return text.replace("|", r"\|")


def render_markdown() -> str:
    """Render the whole channel map as Markdown, for inclusion in a paper.

    Returns
    -------
    str
        A Markdown document ending in exactly one newline: a heading, a
        provenance line naming ``arXiv:2604.05897v1``, the summary table over
        :data:`_COLUMNS`, and one per-channel block carrying the continuum and
        discretized LaTeX.

    Raises
    ------
    AttributeError
        If :data:`_COLUMNS` names an attribute :class:`ChannelEquation` does not
        have.

    Notes
    -----
    The per-channel equation blocks are separate from the table because LaTeX
    maths does not fit legibly in a table cell -- and shrinking the equations to
    make them fit is exactly how a transcription error survives review.

    Exactly one trailing newline is emitted, because
    ``tests/test_equation_map.py`` compares this output byte-for-byte against
    the committed document to catch a stale checked-in copy.

    Examples
    --------
    >>> doc = render_markdown()
    >>> doc.splitlines()[0]
    '# Equation map'
    >>> "arXiv:2604.05897v1" in doc
    True
    >>> doc.endswith(chr(10)) and not doc.endswith(chr(10) * 2)
    True
    >>> all(name in doc for name in CHANNEL_EQUATIONS)
    True
    """
    ref = f"arXiv:{_ARXIV_ID}{_ARXIV_VERSION}"
    lines = [
        "# Equation map",
        "",
        f"Generated from `simulator/equation_map.py`. Reference: {ref} "
        "(Herr, Tikan & Kippenberg, 7 Apr 2026); equation and section numbers "
        "are v1 numbers.",
        "",
        f"`{_EMPTY}` means the repository does not record that cross-reference. "
        "It is left blank rather than guessed.",
        "",
        "## Channel summary",
        "",
    ]

    headers = [header for _, header in _COLUMNS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for entry in CHANNEL_EQUATIONS.values():
        cells = [
            _md_escape(_cell(getattr(entry, attr))) for attr, _ in _COLUMNS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Equations", ""]
    for entry in CHANNEL_EQUATIONS.values():
        where = ", ".join(x for x in (entry.paper_section, entry.paper_eq) if x)
        heading = f"### `{entry.channel}`"
        if where:
            heading += f" ({where})"
        lines += [
            heading,
            "",
            f"Implemented by `{entry.code_symbol}` in `{entry.code_file}`.",
            "",
            "Continuum:",
            "",
            "```latex",
            entry.equation_latex,
            "```",
            "",
            "Discretized (as implemented):",
            "",
            "```latex",
            entry.discrete_latex,
            "```",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def _latex_escape(text: str) -> str:
    """Escape a plain-text cell for LaTeX (never applied to the maths fields).

    ``|`` becomes ``\\textbar{}`` rather than passing through: under the
    default OT1 font encoding a bare ``|`` typesets as an em dash, which would
    turn the ``colour`` column's ``a | b | c`` alternatives into ``a -- b -- c``
    in a printed table without any error to notice.

    The backslash must be escaped FIRST (otherwise it would re-escape the
    backslashes introduced below), and the ``\\text...{}`` replacements must
    come AFTER the brace escaping so their own braces stay live.
    """
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return (
        text.replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
        .replace("|", r"\textbar{}")
        .replace("<", r"\textless{}")
        .replace(">", r"\textgreater{}")
    )


def render_latex() -> str:
    """Render the channel map as a booktabs LaTeX table.

    Returns
    -------
    str
        A complete ``table`` environment ending in one newline, with a
        generated-file banner, a caption citing ``arXiv:2604.05897v1`` and one
        row per channel.

    Raises
    ------
    AttributeError
        If a column names an attribute :class:`ChannelEquation` does not have.

    Notes
    -----
    Requires ``\\usepackage{booktabs}`` in the document preamble. Only a compact
    subset of the columns is emitted: the full set does not fit a page width,
    and the maths lives in :func:`render_markdown` and in this module.

    Plain-text cells go through ``_latex_escape``, never the maths fields. That
    escaping matters more than it looks: under the default OT1 font encoding a
    bare ``|`` typesets as an em dash, which would silently turn the ``colour``
    column's ``a | b | c`` alternatives into ``a -- b -- c`` in a printed table
    with no error to notice.

    Examples
    --------
    >>> tex = render_latex()
    >>> tex.splitlines()[0]
    '% Generated by simulator/equation_map.py -- do not edit by hand.'
    >>> tex.count(chr(92) + "toprule"), tex.count(chr(92) + "bottomrule")
    (1, 1)
    >>> "arXiv:2604.05897v1" in tex
    True
    """
    columns = (
        ("channel", "Channel"),
        ("paper_eq", "Eq."),
        ("paper_section", "Sec."),
        ("enters_as", "Enters as"),
        ("colour", "Colour"),
        ("shares_random_source_with", "Shares source"),
    )
    lines = [
        r"% Generated by simulator/equation_map.py -- do not edit by hand.",
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Noise channels of the stochastic LLE benchmark, the "
        rf"equations they implement (arXiv:{_ARXIV_ID}{_ARXIV_VERSION}), and "
        r"the random source each one draws from.}",
        r"  \label{tab:equation-map}",
        r"  \begin{tabular}{" + "l" * len(columns) + "}",
        r"    \toprule",
        "    " + " & ".join(header for _, header in columns) + r" \\",
        r"    \midrule",
    ]
    for entry in CHANNEL_EQUATIONS.values():
        cells = [
            _latex_escape(_cell(getattr(entry, attr))) for attr, _ in columns
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - manual regeneration
    import sys

    # write, not print: render_markdown() already ends with exactly one
    # newline, and print would add a second one that the staleness test in
    # tests/test_equation_map.py would then flag.
    sys.stdout.write(render_markdown())
