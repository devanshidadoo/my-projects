"""A marketplace fulfilment generator, calibrated to Olist's marginals.

Why generate at all
-------------------
The policy this project builds is counterfactual: it asks *what would have happened to this
order had we nudged the seller / bought express*. No operational log answers that. A log records
the delivery that happened under whatever was done, and delivery is not re-runnable. Every
intervention study therefore either runs an experiment or assumes an uplift -- and the second
one, done offline, cannot be checked.

So the generator carries the latent structure and replays it under each action with common
random numbers. For every order it knows ``Y(a)`` for all four actions, which makes the policy
comparison in ``evaluation/decisions.py`` exact rather than assumed. The real-data path
(:mod:`deliveryrisk.data.olist`) runs everything except that comparison, and says so.

The process
-----------
An order's delivery time decomposes into two mechanisms that different interventions act on::

    delivered = approved + H + T
        H  handling: seller picks, packs, hands over.   nudging the seller compresses this.
        T  transit:  carrier moves the parcel.          buying express compresses this.

    late  <=>  H + T  >  slack,   slack = promise - approved

Both are congested rather than i.i.d., and the congestion is the part worth learning:

* **Seller backlog.** A seller with many approved-but-not-collected orders is slower on the next
  one. Computed from the order stream and fed back into handling time.
* **Carrier day-load.** A carrier running above its daily capacity adds transit days to every
  parcel it touches that day -- so the risk of an order depends on other orders it has nothing
  to do with, which is exactly why a per-order static feature set is not enough.
* **Incidents.** Episodic disruption on a (carrier, region) pair for a week or two: a strike, a
  depot failure, a flood. These make the *history* features non-stationary, which is what makes
  a seller's or lane's past late rate an imperfect guide to its present.

Calibration targets Olist's published marginals (see ``docs/METHODOLOGY.md`` §2); the promise
buffer is solved, not guessed, so the generated late rate hits the configured target exactly.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DAY = 86_400.0
HOUR = 3_600.0

#: The four actions the policy may take at approval time. ``none`` is doing nothing, which is
#: also a decision and is priced like one.
ACTIONS: tuple[str, ...] = ("none", "nudge", "expedite", "both")

#: Brazilian-style state codes with an SP-dominant popularity skew, as in Olist.
STATES: tuple[str, ...] = (
    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", "PE", "CE", "PA", "MT",
    "MA", "MS", "PB", "PI", "RN", "AL", "SE", "TO", "RO", "AM", "AC", "AP", "RR",
)
STATE_WEIGHTS = np.array(
    [42.0, 12.9, 11.7, 5.4, 5.0, 3.6, 3.4, 2.1, 2.0, 2.0, 1.7, 1.3, 1.0, 0.9,
     0.8, 0.7, 0.5, 0.5, 0.5, 0.4, 0.3, 0.3, 0.25, 0.15, 0.08, 0.07, 0.05]
)

CATEGORIES: tuple[str, ...] = (
    "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories",
    "housewares", "watches_gifts", "telephony", "garden_tools", "auto", "toys", "cool_stuff",
    "perfumery", "baby", "electronics", "stationery", "fashion_bags", "pet_shop", "office_furniture",
    "consoles_games", "luggage_accessories", "construction_tools", "musical_instruments", "books",
    "food_drink", "small_appliances", "industry_commerce",
)

PAYMENT_TYPES = ("credit_card", "boleto", "voucher", "debit_card")
PAYMENT_WEIGHTS = np.array([0.7392, 0.1904, 0.0554, 0.0150])


@dataclass
class SyntheticConfig:
    """Every knob of the generator. Nothing in the process reads a constant that is not here."""

    # ---- scale
    n_orders: int = 99_441
    n_unique_customers: int = 96_096   # one `customer_id` per order, as in Olist; this is people
    n_sellers: int = 3_095
    n_products: int = 32_951
    span_days: float = 730.0
    #: Days between the last purchase and the extract. Orders still moving at the extract are
    #: right-censored: they appear as `shipped` with a null `delivered_ts`, exactly as a live
    #: extract would show them. Too short a lag censors the slow orders preferentially and
    #: quietly deflates the observed late rate.
    extract_lag_days: float = 45.0
    seed: int = 7

    # ---- catalogue and pricing
    price_log_mu: float = 4.33          # exp(4.33) ~ 76 median line price
    price_log_sigma: float = 0.97
    freight_base: float = 5.2
    freight_per_kg: float = 3.4
    freight_per_1000km: float = 4.6
    items_extra_lambda: float = 0.133   # Poisson mean of extra lines beyond the first

    # ---- handling (seller side)
    handling_floor_days: float = 0.35   # physical minimum: nothing ships in zero time
    handling_mu_merchant: float = 0.62  # lognormal mu of the seller's own base handling, in days
    handling_mu_warehouse: float = 0.10
    handling_sigma: float = 0.70
    seller_quality_sigma: float = 0.42  # latent, never observed; only its consequences are
    # A minority of lines stall: the seller is out of stock, the pick fails, nobody looks at the
    # queue until Monday. This is the tail that makes handling a *cause* of lateness rather than
    # a rounding error against transit, and it is the tail a nudge acts on.
    stall_share: float = 0.08
    stall_median_days: float = 5.0
    stall_log_sigma: float = 1.2        # heavy: a stalled line is occasionally stalled for weeks
    backlog_congestion: float = 0.090   # extra handling days per backlogged order
    backlog_cap: float = 40.0
    weekend_handling_penalty: float = 1.10  # days added when handover falls on a weekend

    # ---- demand shape
    growth_start: float = 0.80          # relative daily volume at the start of the window
    growth_end: float = 1.25            # ... and at the end. Kept mild on purpose: a strong
    #: secular trend in volume turns into a secular trend in the late rate, and then a
    #: chronological split is measuring the trend rather than the model.
    peak_amplitude: float = 1.5         # extra relative volume at the November peaks
    capacity_headroom: float = 1.8      # carrier daily capacity as a multiple of mean volume

    # ---- transit (carrier side)
    transit_floor_share: float = 0.34   # express cannot beat physics: floor as a share of base
    transit_sigma: float = 0.52
    transit_speed_scale: float = 0.50   # <1 slows every lane; calibrates mean delivery days
    capacity_congestion: float = 0.85   # transit-day multiplier slope above 100 % carrier load
    incident_rate_per_carrier_year: float = 3.5
    incident_days_mu: float = 9.0
    incident_transit_multiplier: float = 1.85

    # ---- the promise
    #: The quote is a rule of thumb, not a forecast: advertised handling, a multiple of the
    #: lane's published transit time, and a flat buffer. The multiple is what makes slack scale
    #: with distance, so a short lane has little absolute room and a long one has plenty.
    promise_transit_multiple: float = 1.8
    promise_buffer_days: float | None = None  # solved from target_late_rate when None
    promise_noise_days: float = 2.2
    target_late_rate: float = 0.079     # Olist: share of delivered orders past the promise

    # ---- satisfaction (priced in policy/economics.py, never used as a feature)
    review_base_score: float = 4.42
    review_late_penalty: float = 0.42
    review_noise: float = 0.92

    # ---- lifecycle
    approval_lag_hours_card: float = 0.35
    approval_lag_hours_boleto: float = 30.0
    cancelled_share: float = 0.021

    # ---- intervention physics (the truth the policy is trying to exploit)
    nudge_handling_reduction: float = 0.55   # share of *excess* handling removed, x compliance
    nudge_compliance_merchant: float = 0.72
    nudge_compliance_warehouse: float = 0.30  # a warehouse is already running at its floor
    expedite_transit_reduction: float = 0.42  # share of transit above the floor removed

    def replace(self, **kw) -> SyntheticConfig:
        d = asdict(self)
        d.update(kw)
        return SyntheticConfig(**d)


@dataclass
class GeneratedData:
    """The nine tables, plus the truth that no database contains.

    Attributes
    ----------
    frames:
        The nine operational tables, ready for :meth:`deliveryrisk.data.db.Database.load_frames`.
    truth:
        One row per order: the latent delay components, the counterfactual lateness under each
        action, and the cause attribution. **Never loaded into the database**, because a real
        extract cannot contain it -- it is the evaluation oracle only.
    """

    frames: dict[str, pd.DataFrame]
    truth: pd.DataFrame
    config: SyntheticConfig = field(default_factory=SyntheticConfig)
    #: One row per generated carrier disruption. Diagnostic only -- nothing downstream reads it,
    #: and a real extract would not contain it either.
    incidents: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def orders(self) -> pd.DataFrame:
        return self.frames["orders"]


# ---------------------------------------------------------------------------- helpers
def _ids(prefix: str, n: int) -> np.ndarray:
    width = max(6, len(str(n)))
    return np.array([f"{prefix}{i:0{width}d}" for i in range(n)], dtype=object)


def _lognormal(rng: np.random.Generator, mu: float, sigma: float, n: int) -> np.ndarray:
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def _weighted_choice(rng: np.random.Generator, n: int, weights: np.ndarray) -> np.ndarray:
    p = weights / weights.sum()
    return rng.choice(len(weights), size=n, p=p)


def _zipf_popularity(rng: np.random.Generator, n: int, alpha: float = 1.05) -> np.ndarray:
    """Popularity weights with a heavy head and a long tail, as marketplaces have."""
    ranks = np.arange(1, n + 1)
    w = 1.0 / ranks**alpha
    rng.shuffle(w)
    return w / w.sum()


# ---------------------------------------------------------------------------- entities
def _make_carriers(rng: np.random.Generator) -> pd.DataFrame:
    spec = [
        # name,           service,    reliability, surcharge, capacity/day
        ("norte_log",     "economy",  0.86, 11.0, 240),
        ("via_rapida",    "standard", 0.92, 14.5, 520),
        ("correios_br",   "standard", 0.90, 12.0, 900),
        ("aereo_express",  "express", 0.96, 24.0, 180),
    ]
    return pd.DataFrame(
        {
            "carrier_id": [f"CAR{i:03d}" for i in range(len(spec))],
            "carrier_name": [s[0] for s in spec],
            "service_level": [s[1] for s in spec],
            "reliability_index": [s[2] for s in spec],
            "expedite_surcharge": [s[3] for s in spec],
            "daily_capacity": [s[4] for s in spec],
        }
    )


def _state_coords(rng: np.random.Generator) -> np.ndarray:
    """A plane embedding of the states, so lane distance is a consistent metric."""
    angles = np.linspace(0, 2 * np.pi, len(STATES), endpoint=False)
    radius = 400 + 1800 * np.sqrt(np.linspace(0.02, 1.0, len(STATES)))
    xy = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return xy + rng.normal(0, 60, size=xy.shape)


def _make_lanes(
    rng: np.random.Generator, carriers: pd.DataFrame, speed_scale: float = 1.0
) -> pd.DataFrame:
    coords = _state_coords(rng)
    rows = []
    n_states = len(STATES)
    for o in range(n_states):
        for d in range(n_states):
            dist = float(np.hypot(*(coords[o] - coords[d])))
            # Two carriers serve each lane: one cheap, one reliable. Long lanes also get air.
            options = [1, 2] if dist < 1500 else [0, 2]
            if dist > 2200 or rng.random() < 0.22:
                options.append(3)
            for ci in options:
                svc = carriers.service_level.iloc[ci]
                speed = {"economy": 330.0, "standard": 460.0, "express": 900.0}[svc] * speed_scale
                base = 1.1 + dist / speed + (0.8 if svc == "economy" else 0.35)
                rows.append(
                    {
                        "lane_id": f"LN{len(rows):05d}",
                        "origin_state": STATES[o],
                        "dest_state": STATES[d],
                        "carrier_id": carriers.carrier_id.iloc[ci],
                        "distance_km": round(dist, 1),
                        "base_transit_days": round(base, 3),
                    }
                )
    return pd.DataFrame(rows)


def _make_sellers(rng: np.random.Generator, cfg: SyntheticConfig) -> pd.DataFrame:
    n = cfg.n_sellers
    state_idx = _weighted_choice(rng, n, STATE_WEIGHTS)
    mode = np.where(rng.random(n) < 0.22, "warehouse", "merchant")
    # Sellers join over the whole window, so "new seller with no history" is a real, common state.
    onboarded = np.sort(rng.random(n)) ** 1.4 * cfg.span_days * DAY * 0.82
    onboarded[: int(0.35 * n)] = 0.0  # a third were there on day one
    return pd.DataFrame(
        {
            "seller_id": _ids("SEL", n),
            "seller_state": [STATES[i] for i in state_idx],
            "seller_zip_prefix": rng.integers(1000, 99999, n),
            "onboarded_ts": onboarded,
            "fulfilment_mode": mode,
        }
    )


def _make_products(rng: np.random.Generator, cfg: SyntheticConfig) -> pd.DataFrame:
    n = cfg.n_products
    cat = rng.choice(len(CATEGORIES), size=n, p=_zipf_popularity(rng, len(CATEGORIES), 0.7))
    # Bulk is category-driven: furniture is heavy, books are not.
    bulk = np.array([1.0 + 2.6 * ((i * 7919) % 11) / 10.0 for i in range(len(CATEGORIES))])
    weight = _lognormal(rng, 6.4, 1.15, n) * bulk[cat]
    return pd.DataFrame(
        {
            "product_id": _ids("PRD", n),
            "category": [CATEGORIES[i] for i in cat],
            "weight_g": np.round(weight, 1),
            "length_cm": np.round(8 + 40 * rng.random(n) * np.sqrt(bulk[cat]), 1),
            "height_cm": np.round(4 + 26 * rng.random(n) * np.sqrt(bulk[cat]), 1),
            "width_cm": np.round(6 + 30 * rng.random(n) * np.sqrt(bulk[cat]), 1),
        }
    )


def _make_customers(rng: np.random.Generator, cfg: SyntheticConfig, n_orders: int) -> pd.DataFrame:
    """One row per order, as in the source schema: `customer_id` is an order-scoped shipping
    identity and `customer_unique_id` is the person. Repeat buyers are the ~3 % whose unique id
    appears more than once, and they are the only rows for which a customer-history feature has
    anything in it.
    """
    n_unique = min(cfg.n_unique_customers, n_orders)
    unique_of_row = np.empty(n_orders, dtype=np.int64)
    unique_of_row[:n_unique] = np.arange(n_unique)
    if n_orders > n_unique:
        unique_of_row[n_unique:] = rng.integers(0, n_unique, n_orders - n_unique)
    unique_of_row = rng.permutation(unique_of_row)
    unique_ids = _ids("CUQ", n_unique)
    # State is a property of the person, not of the order.
    unique_state = _weighted_choice(rng, n_unique, STATE_WEIGHTS)[unique_of_row]
    return pd.DataFrame(
        {
            "customer_id": _ids("CUS", n_orders),
            "customer_unique_id": unique_ids[unique_of_row],
            "customer_state": [STATES[i] for i in unique_state],
            "customer_zip_prefix": rng.integers(1000, 99999, n_orders),
            "first_seen_ts": np.zeros(n_orders),  # replaced with the row's own purchase time
        }
    )


# ---------------------------------------------------------------------------- order stream
def _purchase_times(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    """Purchase timestamps with growth, a weekly cycle, and two shopping peaks per year.

    The peaks matter more than they look: they are when carrier capacity binds, so they are when
    a static per-order feature set is least able to explain who is late.
    """
    n_days = int(cfg.span_days)
    day = np.arange(n_days)
    growth = cfg.growth_start + (cfg.growth_end - cfg.growth_start) * day / max(n_days - 1, 1)
    weekly = 1.0 + 0.22 * np.cos(2 * np.pi * (day % 7) / 7.0)
    peaks = np.ones(n_days)
    for anchor in (328.0, 693.0):  # late-November shopping events
        peaks += cfg.peak_amplitude * np.exp(-0.5 * ((day - anchor) / 4.5) ** 2)
    for anchor in (352.0, 717.0):  # the Christmas shoulder
        peaks += 0.35 * cfg.peak_amplitude * np.exp(-0.5 * ((day - anchor) / 9.0) ** 2)
    intensity = growth * weekly * peaks
    p = intensity / intensity.sum()
    days = rng.choice(n_days, size=cfg.n_orders, p=p)
    # Hour of day: a broad afternoon/evening shopping profile.
    hours = np.clip(rng.normal(15.0, 4.6, cfg.n_orders), 0, 23.999)
    ts = days * DAY + hours * HOUR
    return np.sort(ts)


def _lane_lookup(lanes: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, grp in lanes.groupby(["origin_state", "dest_state"], sort=False):
        out[(str(key[0]), str(key[1]))] = grp.index.to_numpy()
    return out


def _choose_lanes(
    rng: np.random.Generator,
    lanes: pd.DataFrame,
    carriers: pd.DataFrame,
    origin: np.ndarray,
    dest: np.ndarray,
    weight_g: np.ndarray,
) -> np.ndarray:
    """Pick a lane per item: heavy parcels skew to the cheap carrier, light ones to the fast one."""
    lookup = _lane_lookup(lanes)
    svc = carriers.set_index("carrier_id").service_level.to_dict()
    lane_service = lanes.carrier_id.map(svc).to_numpy()
    out = np.empty(len(origin), dtype=np.int64)
    pairs = pd.DataFrame({"o": origin, "d": dest, "w": weight_g})
    for (o, d), grp in pairs.groupby(["o", "d"], sort=False):
        cand = lookup[(str(o), str(d))]
        svc_c = lane_service[cand]
        heavy = grp.w.to_numpy() > 4000.0
        base = np.where(svc_c == "economy", 1.6, np.where(svc_c == "standard", 3.0, 0.45))
        heavy_w = np.where(svc_c == "economy", 3.4, np.where(svc_c == "standard", 2.4, 0.12))
        light_p = base / base.sum()
        heavy_p = heavy_w / heavy_w.sum()
        picks = np.where(
            heavy,
            rng.choice(len(cand), size=len(grp), p=heavy_p),
            rng.choice(len(cand), size=len(grp), p=light_p),
        )
        out[grp.index.to_numpy()] = cand[picks]
    return out


def _incident_multiplier(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    carrier_idx: np.ndarray,
    dest_state_idx: np.ndarray,
    day: np.ndarray,
    n_carriers: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Episodic (carrier, region) disruption: the reason history decays as a predictor."""
    years = cfg.span_days / 365.0
    mult = np.ones(len(day))
    rows = []
    for c in range(n_carriers):
        n_inc = rng.poisson(cfg.incident_rate_per_carrier_year * years)
        for _ in range(int(n_inc)):
            start = rng.random() * cfg.span_days
            dur = max(2.0, rng.exponential(cfg.incident_days_mu))
            region = rng.choice(len(STATES), size=rng.integers(2, 7), replace=False)
            hit = (
                (carrier_idx == c)
                & (day >= start)
                & (day < start + dur)
                & np.isin(dest_state_idx, region)
            )
            severity = cfg.incident_transit_multiplier * rng.uniform(0.75, 1.3)
            mult[hit] *= severity
            rows.append(
                {
                    "carrier_idx": c,
                    "start_day": start,
                    "duration_days": dur,
                    "n_states": len(region),
                    "multiplier": severity,
                    "items_hit": int(hit.sum()),
                }
            )
    return mult, pd.DataFrame(rows)


def _backlog_at(approved: np.ndarray, pickup: np.ndarray, key: np.ndarray) -> np.ndarray:
    """For each row, how many rows with the same key were open (approved, not yet picked up).

    Counted strictly before the row's own approval instant, so a row never counts itself. Done by
    sorting once per key rather than looping: ``searchsorted`` over the sorted approval and
    pickup times of the group gives ``opened_before - closed_before``.
    """
    out = np.zeros(len(approved), dtype=np.float64)
    order = np.argsort(key, kind="stable")
    keys_sorted = key[order]
    bounds = np.flatnonzero(np.r_[True, keys_sorted[1:] != keys_sorted[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = order[lo:hi]
        a = approved[rows]
        p = np.where(np.isnan(pickup[rows]), np.inf, pickup[rows])
        a_sorted = np.sort(a)
        p_sorted = np.sort(p)
        opened = np.searchsorted(a_sorted, a, side="left")
        closed = np.searchsorted(p_sorted, a, side="right")
        out[rows] = np.maximum(opened - closed, 0)
    return out


# ---------------------------------------------------------------------------- the process
def generate(cfg: SyntheticConfig | None = None) -> GeneratedData:
    """Generate the nine tables and the counterfactual truth."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    horizon = (cfg.span_days + cfg.extract_lag_days) * DAY  # when the extract is taken

    carriers = _make_carriers(rng)
    # Capacity is expressed relative to the configured volume, so a smoke run congests like the
    # headline run instead of never congesting at all.
    share = np.array([0.15, 0.30, 0.45, 0.10])
    per_day = cfg.n_orders / cfg.span_days
    carriers["daily_capacity"] = np.maximum(
        4, np.round(share * per_day * cfg.capacity_headroom)
    ).astype(int)

    lanes = _make_lanes(rng, carriers, cfg.transit_speed_scale)
    sellers = _make_sellers(rng, cfg).sort_values("onboarded_ts", ignore_index=True)
    products = _make_products(rng, cfg)

    n = cfg.n_orders
    purchase = _purchase_times(rng, cfg)
    customers = _make_customers(rng, cfg, n)
    customers["first_seen_ts"] = purchase
    cust_idx = np.arange(n)  # orders and customer rows are 1:1, as in the source schema
    order_ids = _ids("ORD", n)

    # ---- items -------------------------------------------------------------------
    n_items = 1 + np.minimum(rng.poisson(cfg.items_extra_lambda, n), 5)
    item_order = np.repeat(np.arange(n), n_items)
    item_seq = np.concatenate([np.arange(k) + 1 for k in n_items])
    m = len(item_order)

    prod_idx = rng.choice(cfg.n_products, size=m, p=_zipf_popularity(rng, cfg.n_products, 0.85))
    # Sellers are chosen among those already onboarded at purchase time, skewed to the incumbents.
    onboarded_sorted = sellers.onboarded_ts.to_numpy()
    eligible = np.searchsorted(onboarded_sorted, purchase[item_order], side="right")
    eligible = np.maximum(eligible, 1)
    sel_idx = np.floor(eligible * rng.random(m) ** 1.8).astype(np.int64)
    sel_idx = np.minimum(sel_idx, eligible - 1)

    weight = products.weight_g.to_numpy()[prod_idx]
    origin = sellers.seller_state.to_numpy()[sel_idx]
    dest = customers.customer_state.to_numpy()[cust_idx][item_order]
    lane_idx = _choose_lanes(rng, lanes, carriers, origin, dest, weight)
    distance = lanes.distance_km.to_numpy()[lane_idx]
    base_transit = lanes.base_transit_days.to_numpy()[lane_idx]
    carrier_of_lane = lanes.carrier_id.to_numpy()[lane_idx]
    carrier_idx = pd.Index(carriers.carrier_id).get_indexer(carrier_of_lane)
    svc_of_lane = carriers.service_level.to_numpy()[carrier_idx]

    cat_idx = pd.Index(CATEGORIES).get_indexer(products.category.to_numpy()[prod_idx])
    cat_price_factor = 0.62 + 0.86 * ((np.arange(len(CATEGORIES)) * 37 % 13) / 12.0)
    price = np.round(
        _lognormal(rng, cfg.price_log_mu, cfg.price_log_sigma, m) * cat_price_factor[cat_idx], 2
    )
    freight = np.round(
        cfg.freight_base
        + cfg.freight_per_kg * weight / 1000.0
        + cfg.freight_per_1000km * distance / 1000.0
        + np.where(svc_of_lane == "express", 6.5, 0.0)
        + rng.normal(0, 1.2, m),
        2,
    ).clip(min=3.0)

    # ---- payment and approval ----------------------------------------------------
    pay_idx = rng.choice(len(PAYMENT_TYPES), size=n, p=PAYMENT_WEIGHTS)
    pay_type = np.array(PAYMENT_TYPES)[pay_idx]
    lag_hours = np.where(
        pay_type == "boleto",
        rng.exponential(cfg.approval_lag_hours_boleto, n),
        rng.exponential(cfg.approval_lag_hours_card, n),
    )
    approved = purchase + lag_hours * HOUR
    approved_item = approved[item_order]

    # ---- handling: seller base, then backlog congestion ---------------------------
    mode = sellers.fulfilment_mode.to_numpy()
    seller_quality = rng.normal(0, cfg.seller_quality_sigma, cfg.n_sellers)  # latent, unobserved
    seller_mu = np.where(mode == "warehouse", cfg.handling_mu_warehouse, cfg.handling_mu_merchant)
    compliance = np.where(
        mode == "warehouse",
        rng.beta(3, 7, cfg.n_sellers) * (2 * cfg.nudge_compliance_warehouse),
        rng.beta(6, 3, cfg.n_sellers) * (1.45 * cfg.nudge_compliance_merchant),
    ).clip(0.0, 1.0)

    handling_raw = cfg.handling_floor_days + _lognormal(
        rng, 0.0, cfg.handling_sigma, m
    ) * np.exp(seller_mu[sel_idx] + seller_quality[sel_idx])
    # Bulky parcels take longer to pack, and a multi-line order queues behind itself.
    handling_raw *= 1.0 + 0.10 * (weight > 6000) + 0.06 * (n_items[item_order] - 1)
    stalled = rng.random(m) < cfg.stall_share
    handling_raw = handling_raw + stalled * _lognormal(
        rng, np.log(cfg.stall_median_days), cfg.stall_log_sigma, m
    )

    pickup_guess = approved_item + handling_raw * DAY
    backlog = _backlog_at(approved_item, pickup_guess, sellers.seller_id.to_numpy()[sel_idx])
    handling = handling_raw + cfg.backlog_congestion * np.minimum(backlog, cfg.backlog_cap)
    handover_dow = np.floor((approved_item + handling * DAY) / DAY).astype(np.int64) % 7
    handling = handling + cfg.weekend_handling_penalty * np.isin(handover_dow, (5, 6))
    pickup = approved_item + handling * DAY

    # ---- transit: carrier load, incidents, lane noise ------------------------------
    pickup_day = np.floor(pickup / DAY).astype(np.int64)
    n_days = int(cfg.span_days) + 30
    load = np.zeros((len(carriers), n_days))
    for c in range(len(carriers)):
        hit = carrier_idx == c
        load[c] = np.bincount(np.clip(pickup_day[hit], 0, n_days - 1), minlength=n_days)
    cap = carriers.daily_capacity.to_numpy()[:, None]
    load_ratio = load / cap
    congestion = 1.0 + cfg.capacity_congestion * np.clip(load_ratio - 1.0, 0, None)
    cong_item = congestion[carrier_idx, np.clip(pickup_day, 0, n_days - 1)]

    dest_state_idx = pd.Index(STATES).get_indexer(dest)
    incident_mult, incidents = _incident_multiplier(
        rng, cfg, carrier_idx, dest_state_idx, pickup / DAY, len(carriers)
    )
    reliability = carriers.reliability_index.to_numpy()[carrier_idx]
    transit_noise = _lognormal(rng, 0.0, cfg.transit_sigma, m) * (1.0 + (0.93 - reliability))
    transit = base_transit * transit_noise * cong_item * incident_mult
    transit_floor = cfg.transit_floor_share * base_transit
    transit = np.maximum(transit, transit_floor)

    delivered_item = pickup + transit * DAY

    # ---- the order's critical path -------------------------------------------------
    crit = (
        pd.DataFrame({"o": item_order, "d": delivered_item})
        .groupby("o", sort=True)["d"]
        .idxmax()
        .to_numpy()
    )
    handling_o = handling[crit]
    transit_o = transit[crit]
    base_transit_o = base_transit[crit]
    transit_floor_o = transit_floor[crit]
    lane_o = lane_idx[crit]
    seller_o = sel_idx[crit]
    delivered = delivered_item[crit]

    # ---- status: cancellations and the in-flight tail ------------------------------
    # Decided before the promise is set, because neither depends on it -- and because the late
    # rate has to be solved on the orders an extract would actually show as delivered.
    status = np.array(["delivered"] * n, dtype=object)
    cancelled = rng.random(n) < cfg.cancelled_share
    status[cancelled] = "cancelled"
    in_flight = (~cancelled) & (delivered > horizon)
    status[in_flight] = "shipped"
    observable = status == "delivered"

    # ---- the promise, solved to hit the target late rate ---------------------------
    advertised_handling = np.where(mode == "warehouse", 1.0, 2.0)[seller_o]
    promise_core = (
        advertised_handling
        + cfg.promise_transit_multiple * base_transit_o
        + rng.normal(0, cfg.promise_noise_days, n)
    )
    if cfg.promise_buffer_days is None:
        buffer = _solve_buffer(promise_core, purchase, approved, handling_o, transit_o,
                               cfg.target_late_rate, mask=observable)
        log.info("solved promise buffer = %.2f days for target late rate %.3f",
                 buffer, cfg.target_late_rate)
    else:
        buffer = float(cfg.promise_buffer_days)
    promise_days = np.ceil(np.maximum(promise_core + buffer, 2.0))
    estimated_delivery = purchase + promise_days * DAY

    # ---- counterfactual replay -----------------------------------------------------
    comp_item = compliance[sel_idx]
    h_by_action = {
        "none": handling,
        "nudge": cfg.handling_floor_days
        + (handling - cfg.handling_floor_days)
        * (1.0 - cfg.nudge_handling_reduction * comp_item),
    }
    h_by_action["expedite"] = h_by_action["none"]
    h_by_action["both"] = h_by_action["nudge"]
    t_by_action = {
        "none": transit,
        "expedite": transit_floor + (transit - transit_floor) * (1.0 - cfg.expedite_transit_reduction),
    }
    t_by_action["nudge"] = t_by_action["none"]
    t_by_action["both"] = t_by_action["expedite"]

    slack_days = (estimated_delivery - approved) / DAY
    late_by_action: dict[str, np.ndarray] = {}
    lateness_days: dict[str, np.ndarray] = {}
    for a in ACTIONS:
        done = approved_item + (h_by_action[a] + t_by_action[a]) * DAY
        order_done = (
            pd.DataFrame({"o": item_order, "d": done}).groupby("o", sort=True)["d"].max().to_numpy()
        )
        days_over = (order_done - estimated_delivery) / DAY
        late_by_action[a] = days_over > 0
        lateness_days[a] = days_over
    late = late_by_action["none"]

    pickup_o = np.where(cancelled, np.nan, approved + handling_o * DAY)
    delivered_col = np.where(cancelled | in_flight, np.nan, delivered)
    pickup_o = np.where(in_flight & (pickup_o > horizon), np.nan, pickup_o)

    frames = _assemble_frames(
        rng, cfg, carriers, lanes, sellers, products, customers,
        order_ids=order_ids, cust_idx=cust_idx, status=status, purchase=purchase,
        approved=approved, pickup=pickup_o, delivered=delivered_col,
        estimated=estimated_delivery, item_order=item_order, item_seq=item_seq,
        prod_idx=prod_idx, sel_idx=sel_idx, lane_idx=lane_idx, price=price, freight=freight,
        advertised_handling=np.where(mode == "warehouse", 1.0, 2.0)[sel_idx],
        pay_type=pay_type, rng_lateness=lateness_days["none"],
    )

    truth = pd.DataFrame(
        {
            "order_id": order_ids,
            "seller_id": sellers.seller_id.to_numpy()[seller_o],
            "lane_id": lanes.lane_id.to_numpy()[lane_o],
            "carrier_id": lanes.carrier_id.to_numpy()[lane_o],
            "approved_ts": approved,
            "slack_days": slack_days,
            "handling_days": handling_o,
            "transit_days": transit_o,
            "base_transit_days": base_transit_o,
            "handling_floor_days": cfg.handling_floor_days,
            "transit_floor_days": transit_floor_o,
            "seller_compliance": compliance[seller_o],
            "status": status,
            "observable": status == "delivered",
        }
    )
    for a in ACTIONS:
        truth[f"late_{a}"] = late_by_action[a]
        truth[f"days_over_{a}"] = lateness_days[a]
    # Cause attribution. Both columns are functions of *observed* timestamps (handling, transit,
    # slack) plus two operational constants -- so the real-data path can compute them too. What
    # the real path cannot compute is whether an intervention would have delivered the
    # compression; that is the assumed uplift, and `configs/uplift_sensitivity.yaml` sweeps it.
    truth["handling_fixable"] = late & (
        (cfg.handling_floor_days + transit_o) <= slack_days
    )
    truth["transit_fixable"] = late & ((handling_o + transit_floor_o) <= slack_days)
    truth["unfixable"] = late & ~(truth.handling_fixable | truth.transit_fixable)
    truth.attrs["promise_buffer_days"] = float(buffer)
    return GeneratedData(frames=frames, truth=truth, config=cfg, incidents=incidents)


def _solve_buffer(
    promise_core: np.ndarray,
    purchase: np.ndarray,
    approved: np.ndarray,
    handling: np.ndarray,
    transit: np.ndarray,
    target: float,
    *,
    mask: np.ndarray | None = None,
    iters: int = 48,
) -> float:
    """Bisect the promise buffer so the realised late rate equals the target.

    The alternative -- hand-tuning a buffer constant until the late rate looks about right -- is
    how a generator ends up with a late rate nobody can reproduce.
    """
    done = approved + (handling + transit) * 86_400.0
    keep = slice(None) if mask is None else mask

    def late_rate(b: float) -> float:
        promise = purchase + np.ceil(np.maximum(promise_core + b, 2.0)) * 86_400.0
        return float(np.mean((done > promise)[keep]))

    lo, hi = -20.0, 90.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if late_rate(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _assemble_frames(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    carriers: pd.DataFrame,
    lanes: pd.DataFrame,
    sellers: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    **k,
) -> dict[str, pd.DataFrame]:
    """Project the process onto the nine tables -- i.e. throw the latent state away."""
    n = len(k["order_ids"])
    orders = pd.DataFrame(
        {
            "order_id": k["order_ids"],
            "customer_id": customers.customer_id.to_numpy()[k["cust_idx"]],
            "order_status": k["status"],
            "purchase_ts": k["purchase"],
            "approved_ts": k["approved"],
            "pickup_ts": k["pickup"],
            "delivered_ts": k["delivered"],
            "estimated_delivery_ts": k["estimated"],
        }
    )

    items = pd.DataFrame(
        {
            "order_id": k["order_ids"][k["item_order"]],
            "item_seq": k["item_seq"],
            "product_id": products.product_id.to_numpy()[k["prod_idx"]],
            "seller_id": sellers.seller_id.to_numpy()[k["sel_idx"]],
            "lane_id": lanes.lane_id.to_numpy()[k["lane_idx"]],
            "shipping_limit_ts": k["approved"][k["item_order"]]
            + np.ceil(k["advertised_handling"]) * DAY,
            "price": k["price"],
            "freight_value": k["freight"],
        }
    )
    # A cancelled order still had a contractual handover deadline; it just never met one.
    order_total = items.groupby("order_id", sort=False)[["price", "freight_value"]].sum()
    total = (order_total.price + order_total.freight_value).reindex(orders.order_id).to_numpy()

    n_legs = 1 + (rng.random(n) < 0.06).astype(int)
    leg_order = np.repeat(np.arange(n), n_legs)
    leg_seq = np.concatenate([np.arange(c) + 1 for c in n_legs])
    frac = np.where(n_legs[leg_order] == 1, 1.0, np.where(leg_seq == 1, 0.6, 0.4))
    installments = np.where(
        k["pay_type"] == "credit_card",
        np.minimum(1 + rng.poisson(1.9, n), 24),
        1,
    )
    payments = pd.DataFrame(
        {
            "order_id": k["order_ids"][leg_order],
            "payment_seq": leg_seq,
            "payment_type": k["pay_type"][leg_order],
            "installments": installments[leg_order],
            "payment_value": np.round(total[leg_order] * frac, 2),
        }
    )

    # Reviews exist only for delivered orders, and lateness is the dominant term in the score.
    delivered_mask = k["status"] == "delivered"
    days_over = k["rng_lateness"][delivered_mask]
    score = (
        cfg.review_base_score
        - cfg.review_late_penalty * np.clip(days_over, 0, None) ** 0.75
        + rng.normal(0, cfg.review_noise, len(days_over))
    )
    score = np.clip(np.round(score), 1, 5).astype(int)
    created = k["delivered"][delivered_mask] + rng.exponential(1.4, len(days_over)) * DAY
    answered = created + rng.exponential(2.1, len(days_over)) * DAY
    reviews = pd.DataFrame(
        {
            "review_id": _ids("REV", int(delivered_mask.sum())),
            "order_id": k["order_ids"][delivered_mask],
            "review_score": score,
            "review_creation_ts": created,
            "review_answer_ts": np.where(rng.random(len(days_over)) < 0.9, answered, np.nan),
        }
    )

    # A customer's `first_seen_ts` is their first purchase, not an independent draw.
    first_purchase = (
        orders.groupby("customer_id", sort=False).purchase_ts.min().reindex(customers.customer_id)
    )
    customers = customers.copy()
    customers["first_seen_ts"] = first_purchase.fillna(customers.first_seen_ts.mean()).to_numpy()

    return {
        "customers": customers,
        "sellers": sellers,
        "products": products,
        "carriers": carriers,
        "shipping_lanes": lanes,
        "orders": orders,
        "order_items": items,
        "order_payments": payments,
        "order_reviews": reviews,
    }


def dataset_summary(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Marginals for the calibration table. Comparable to the same query on the real files."""
    orders, items = frames["orders"], frames["order_items"]
    delivered = orders[orders.order_status == "delivered"]
    late = delivered.delivered_ts > delivered.estimated_delivery_ts
    pay = frames["order_payments"]
    return {
        "n_orders": float(len(orders)),
        "n_order_items": float(len(items)),
        "n_sellers": float(frames["sellers"].seller_id.nunique()),
        "n_products": float(frames["products"].product_id.nunique()),
        "n_customers_unique": float(frames["customers"].customer_unique_id.nunique()),
        "items_per_order": float(len(items) / len(orders)),
        "mean_price": float(items.price.mean()),
        "median_price": float(items.price.median()),
        "mean_freight": float(items.freight_value.mean()),
        # Per payment *row*, not per order: an order with two legs contributes two rows, which
        # is how the same statistic is defined on the source files.
        "mean_payment_value": float(pay.payment_value.mean()),
        "mean_review_score": float(frames["order_reviews"].review_score.mean()),
        "late_share_delivered": float(late.mean()),
        "mean_delivery_days": float(
            ((delivered.delivered_ts - delivered.purchase_ts) / DAY).mean()
        ),
        "mean_promise_days": float(
            ((delivered.estimated_delivery_ts - delivered.purchase_ts) / DAY).mean()
        ),
        "span_days": float((orders.purchase_ts.max() - orders.purchase_ts.min()) / DAY),
    }
