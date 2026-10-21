"""Weather vocabulary, feature order and physical severity weights."""
# ----------------------------------------------------------------------------
# Constants for parsing wxcodes
# ----------------------------------------------------------------------------

# Descriptors to discard (their modifiers are absorbed into the base phenomena): shallow/partial/patches/low drifting/blowing/freezing
DROP_DESCRIPTORS = {"MI", "PR", "BC", "DR", "BL", "FZ"}
# Intensity prefix -> soft-label weight
INTENSITY_WEIGHT = {"-": 0.5, "": 1.0, "+": 1.5}
VICINITY_WEIGHT = 0.5  # Treat VC (vicinity) as weak intensity

# Continuous weather features (Z-score normalized); vsby is handled separately (capped, without Z-score normalization)
CONT_FEATURES = ["tmpf", "dwpf", "relh", "alti", "u", "v"]
# Final per-timestep feature order (vsby is capped and normalized; flags are not normalized)
FEATURE_ORDER = ["tmpf", "dwpf", "relh", "alti", "vsby", "u", "v",
                 "low_vsby_flag", "wind_var_flag"]

# ---- Fixed class taxonomy (K=9 = 8 weather classes + CLEAR) ----
WEATHER_CLASSES = ["TS", "RA", "SN", "FG", "BR", "HZ", "DUST", "OTHER"]
# Atomic phenomenon -> target class; unknown phenomena map to OTHER
CLASS_MAP = {
    "TS": "TS",
    "RA": "RA", "DZ": "RA", "SH": "RA",
    "SN": "SN", "SG": "SN", "PL": "SN", "GS": "SN", "GR": "SN",
    "FG": "FG",
    "BR": "BR",
    "HZ": "HZ",
    "DU": "DUST", "SA": "DUST", "FU": "DUST",
}
# Severity weights (aviation impact): TS highest, low-visibility SN/FG next, BR/HZ low, CLEAR substantially downweighted
SEVERITY = {"TS": 2.0, "RA": 1.0, "SN": 1.5, "FG": 1.5,
            "BR": 0.7, "HZ": 0.7, "DUST": 1.0, "OTHER": 1.0, "CLEAR": 0.3}
