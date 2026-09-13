"""Generator for an IEEE-CIS-calibrated 18-month transaction stream.

Why a generator at all
----------------------
The closed-loop question ("what happens to a model that only ever sees the labels its own
declines did not suppress?") cannot be answered on a static Kaggle table. It needs *ground
truth for transactions the policy blocked* — counterfactual labels that no real log contains.
A simulator needs an environment whose labels exist for every action, including the ones the
deployed policy refuses to take.

So this module builds a stream whose marginals match IEEE-CIS (volume, fraud rate, amount
distribution, entity cardinalities, class-conditional separability) but whose labels are known
for every transaction. :mod:`clfraud.data.ieee_cis` maps the real Kaggle files onto the same
schema for the parts of the pipeline that do not need counterfactual labels.

Calibration targets (IEEE-CIS ``train_transaction.csv``)
    n = 590,540 transactions; fraud rate 3.499 %; 13,553 distinct ``card1``;
    ``TransactionAmt`` right-skewed, median ~68.8, mean ~135.0, p99 ~1,200;
    5 ``ProductCD`` values with W dominant; span 181.8 days.
The generator keeps all of these except the span, which is stretched to 18 months because the
research question is about *retraining cycles over time* (see ``docs/METHODOLOGY.md``).

Fraud process
-------------
Legit traffic is a non-homogeneous Poisson stream with weekly and daily seasonality. Fraud is
injected as *episodes* drawn from four archetypes whose mixture drifts across the 18 months:

``card_testing``    bursts of small authorisations across many merchants in minutes
``ato_large``       account takeover: 5-20x the card's usual amount, new device, small hours
``merchant_breach`` a compromised merchant leaks cards for a bounded window
``stealth_small``   amounts inside the card's normal range, normal hours, weak signal

The archetypes differ in how strongly they show up in *velocity* features, which is exactly the
property the closed-loop experiment probes: a policy censors the labels of the fraud it catches
best, so the training set's positive class drifts toward whatever the previous model missed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from clfraud.data.schema import validate_frame

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400.0
ARCHETYPES = ("card_testing", "ato_large", "merchant_breach", "stealth_small")


@dataclass
class SyntheticConfig:
    """Knobs for the generator. Defaults reproduce the headline experiment."""

    n_transactions: int = 590_540          # IEEE-CIS train_transaction row count
    n_months: int = 18
    fraud_rate: float = 0.035              # IEEE-CIS: 0.03499
    n_cards: int = 13_553                  # IEEE-CIS distinct card1
    n_merchants: int = 2_400
    n_devices: int = 9_000
    n_addr: int = 330
    seed: int = 20250314

    #: Per-month mixture weights over ARCHETYPES. Linearly interpolated between the first and
    #: last row, which is what produces gradual concept drift.
    #: These are *episode* weights; card_testing episodes emit ~9 transactions each and the
    #: others 1-2, so the transaction-level mix runs roughly 43/18/18/21 -> 22/21/19/38.
    mix_start: tuple[float, ...] = (0.13, 0.24, 0.25, 0.38)
    mix_end: tuple[float, ...] = (0.05, 0.22, 0.20, 0.53)

    #: Merchant category codes. Fraud concentrates in a rotating subset of (mcc, hour-bucket)
    #: cells, which is the part of the signal that can only be learned from *recent* labels.
    n_mcc: int = 40
    n_hour_buckets: int = 4
    #: Hot cells live for this many months, then a fresh draw replaces a share of them.
    hot_cell_count: int = 12
    hot_cell_life_months: int = 3
    #: Share of fraud episodes placed inside currently-hot cells.
    hot_cell_share: float = 0.55
    #: Share of legitimate traffic that arrives in short bursts, so high card velocity is not
    #: on its own a fraud tell (without this the task is trivially separable).
    legit_burst_share: float = 0.11

    #: Number of noisy latent projections emitted as `v*` columns (IEEE-CIS `V*` analogue).
    n_v_columns: int = 20
    #: Of those, how many are pure noise (no latent signal at all).
    n_v_noise: int = 8

    product_codes: tuple[str, ...] = ("W", "C", "R", "H", "S")
    product_weights: tuple[float, ...] = (0.746, 0.116, 0.063, 0.055, 0.020)
    email_domains: tuple[str, ...] = (
        "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
        "comcast.net", "outlook.com", "icloud.com", "live.com", "msn.com",
    )
    email_weights: tuple[float, ...] = (
        0.39, 0.18, 0.11, 0.10, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02,
    )

    _mix_start_arr: np.ndarray = field(init=False, repr=False)
    _mix_end_arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mix_start_arr = np.asarray(self.mix_start, dtype=float)
        self._mix_end_arr = np.asarray(self.mix_end, dtype=float)
        if len(self._mix_start_arr) != len(ARCHETYPES):
            raise ValueError("mix_start must have one weight per archetype")
        if len(self._mix_end_arr) != len(ARCHETYPES):
            raise ValueError("mix_end must have one weight per archetype")

    @property
    def horizon_days(self) -> float:
        return self.n_months * 30.0

    @property
    def n_cells(self) -> int:
        return self.n_mcc * self.n_hour_buckets

    def mixture_at(self, month: int) -> np.ndarray:
        """Archetype mixture in a given month, normalised to sum to one."""
        t = 0.0 if self.n_months <= 1 else month / (self.n_months - 1)
        w = (1.0 - t) * self._mix_start_arr + t * self._mix_end_arr
        return w / w.sum()


class SyntheticStreamGenerator:
    """Draws a labelled transaction stream. Fully determined by ``config.seed``."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.cfg = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------ entities
    def _build_entities(self) -> dict[str, np.ndarray]:
        cfg, rng = self.cfg, self.rng
        # Card activity is heavy-tailed: a few cards transact constantly, most rarely.
        card_activity = rng.pareto(1.6, size=cfg.n_cards) + 1.0
        card_activity /= card_activity.sum()
        # Each card has a typical ticket size; log-normal across the population.
        card_mu = rng.normal(4.25, 0.80, size=cfg.n_cards)     # exp(4.25) ~ $70 median ticket
        card_sigma = rng.uniform(0.45, 1.25, size=cfg.n_cards)

        merchant_pop = rng.pareto(1.25, size=cfg.n_merchants) + 1.0
        merchant_pop /= merchant_pop.sum()
        # Latent per-merchant risk: mostly benign, a right tail of sketchy MCCs.
        merchant_risk = rng.beta(1.4, 9.0, size=cfg.n_merchants)

        # Each card keeps a small habitual merchant basket; 80% of its spend lands there.
        basket_sizes = rng.integers(3, 18, size=cfg.n_cards)
        baskets = [
            rng.choice(cfg.n_merchants, size=int(k), replace=False, p=merchant_pop)
            for k in basket_sizes
        ]
        # Home device per card; fraud often arrives on a device the card has never used.
        # Merchant category. Skewed, like real MCC distributions.
        mcc_w = rng.dirichlet(np.full(cfg.n_mcc, 0.7))
        merchant_mcc = rng.choice(cfg.n_mcc, size=cfg.n_merchants, p=mcc_w)

        card_device = rng.integers(0, cfg.n_devices, size=cfg.n_cards)
        card_addr = rng.integers(0, cfg.n_addr, size=cfg.n_cards)
        card_email = rng.choice(
            len(cfg.email_domains), size=cfg.n_cards, p=np.asarray(cfg.email_weights)
        )
        # Rotating hot (mcc, hour-bucket) cells: month -> array of active cell ids. Cells expire
        # after `hot_cell_life_months`, so a model that cannot refresh its labels goes stale on
        # exactly the component of fraud that moves.
        hot: dict[int, np.ndarray] = {}
        live: list[tuple[int, int]] = []  # (cell_id, expiry_month)
        for m in range(cfg.n_months):
            live = [(c, e) for c, e in live if e > m]
            while len(live) < cfg.hot_cell_count:
                cell = int(rng.integers(0, cfg.n_cells))
                if cell not in {c for c, _ in live}:
                    live.append((cell, m + cfg.hot_cell_life_months))
            hot[m] = np.array([c for c, _ in live], dtype=int)

        mcc_members = [np.flatnonzero(merchant_mcc == k) for k in range(cfg.n_mcc)]

        return {
            "mcc_members": mcc_members,
            "merchant_mcc": merchant_mcc,
            "hot_cells": hot,
            "card_activity": card_activity,
            "card_mu": card_mu,
            "card_sigma": card_sigma,
            "merchant_pop": merchant_pop,
            "merchant_risk": merchant_risk,
            "baskets": baskets,
            "card_device": card_device,
            "card_addr": card_addr,
            "card_email": card_email,
        }

    # ------------------------------------------------------------------ legit stream
    def _legit_timestamps(self, n: int) -> np.ndarray:
        """Non-homogeneous Poisson arrivals with weekly + daily seasonality (thinning)."""
        cfg, rng = self.cfg, self.rng
        horizon = cfg.horizon_days * SECONDS_PER_DAY
        # Oversample uniformly then accept with probability proportional to intensity.
        accepted: list[np.ndarray] = []
        n_have = 0
        while n_have < n:
            draw = rng.uniform(0.0, horizon, size=int((n - n_have) * 2.4) + 1024)
            day = draw / SECONDS_PER_DAY
            hour = (draw % SECONDS_PER_DAY) / 3600.0
            # Daily: trough at 04:00, peak around 13:00-20:00.
            daily = 0.55 + 0.45 * np.sin((hour - 8.0) / 24.0 * 2 * np.pi) + 0.25 * np.exp(
                -0.5 * ((hour - 18.0) / 3.0) ** 2
            )
            weekly = 1.0 + 0.16 * np.sin((day % 7) / 7.0 * 2 * np.pi - 1.0)
            # Mild secular growth in volume across the 18 months.
            trend = 1.0 + 0.28 * (day / cfg.horizon_days)
            intensity = np.clip(daily * weekly * trend, 1e-3, None)
            keep = rng.uniform(0.0, intensity.max(), size=draw.size) < intensity
            accepted.append(draw[keep])
            n_have += int(keep.sum())
        ts = np.concatenate(accepted)[:n]
        ts.sort()
        return ts + SECONDS_PER_DAY  # IEEE-CIS-style offset origin

    def _legit_frame(self, n: int, ent: dict[str, np.ndarray]) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        ts = self._legit_timestamps(n)
        cards = rng.choice(cfg.n_cards, size=n, p=ent["card_activity"])

        # 80% of spend in the card's habitual basket, 20% exploratory across the population.
        merchants = rng.choice(cfg.n_merchants, size=n, p=ent["merchant_pop"])
        in_basket = rng.uniform(size=n) < 0.80
        basket_pick = np.empty(n, dtype=np.int64)
        for i in np.flatnonzero(in_basket):
            b = ent["baskets"][cards[i]]
            basket_pick[i] = b[rng.integers(0, len(b))]
        merchants = np.where(in_basket, basket_pick, merchants)

        # Legit bursts: a share of rows is re-attributed to a recent neighbour's card and
        # timestamp, so "several transactions on one card within the hour" is something
        # legitimate customers produce too (shopping sprees, retries, split tenders). Without
        # this, card velocity alone separates the classes and the task is trivial.
        n_burst = int(cfg.legit_burst_share * n)
        if n_burst:
            anchor = np.sort(rng.choice(np.arange(3, n), size=n_burst, replace=False))
            src = anchor - rng.integers(1, 4, size=n_burst)
            cards[anchor] = cards[src]
            ts[anchor] = ts[src] + rng.exponential(900.0, size=n_burst)
            order = np.argsort(ts, kind="mergesort")
            ts, cards, merchants = ts[order], cards[order], merchants[order]

        amounts = np.exp(rng.normal(ent["card_mu"][cards], ent["card_sigma"][cards]))
        # Round-number ticket sizes are common in legit retail; IEEE-CIS shows the same spike.
        round_mask = rng.uniform(size=n) < 0.18
        amounts = np.where(round_mask, np.maximum(np.round(amounts / 5.0) * 5.0, 1.0), amounts)
        amounts = np.clip(np.round(amounts, 2), 0.5, 32_000.0)

        devices = ent["card_device"][cards].copy()
        # A minority of legit traffic legitimately comes from a new device (travel, new phone).
        new_dev = rng.uniform(size=n) < 0.07
        devices[new_dev] = rng.integers(0, cfg.n_devices, size=int(new_dev.sum()))

        return pd.DataFrame(
            {
                "ts": ts,
                "card_ix": cards,
                "merchant_ix": merchants,
                "device_ix": devices,
                "addr_ix": ent["card_addr"][cards],
                "email_ix": ent["card_email"][cards],
                "amount": amounts,
                "is_fraud": np.zeros(n, dtype=np.int8),
                "archetype": np.full(n, "legit", dtype=object),
            }
        )

    # ------------------------------------------------------------------ fraud episodes
    def _fraud_frame(self, n_fraud_target: int, ent: dict[str, np.ndarray]) -> pd.DataFrame:
        """Draw fraud episodes until the target fraud count is reached."""
        cfg, rng = self.cfg, self.rng
        horizon = cfg.horizon_days * SECONDS_PER_DAY
        rows: list[dict] = []

        # Compromised merchants: each breach is a (merchant, window) pair, drawn up-front so a
        # breach can span a month and be visible in merchant-level aggregates.
        n_breach = 26
        breach_merchants = rng.choice(
            cfg.n_merchants, size=n_breach, replace=False, p=ent["merchant_pop"]
        )
        breach_start = rng.uniform(0.0, horizon * 0.95, size=n_breach)
        breach_len = rng.uniform(6.0, 40.0, size=n_breach) * SECONDS_PER_DAY

        while len(rows) < n_fraud_target:
            t0 = rng.uniform(0.0, horizon)
            month = min(int(t0 / (30.0 * SECONDS_PER_DAY)), cfg.n_months - 1)
            kind = ARCHETYPES[int(rng.choice(len(ARCHETYPES), p=cfg.mixture_at(month)))]
            card = int(rng.choice(cfg.n_cards, p=ent["card_activity"]))
            hot = self._hot_target(month, ent)
            rows.extend(
                self._episode(kind, card, t0, ent, breach_merchants, breach_start, breach_len, hot)
            )

        rows = rows[:n_fraud_target]
        df = pd.DataFrame(rows)
        df["is_fraud"] = np.int8(1)
        return df

    def _hot_target(self, month: int, ent: dict) -> tuple[int, int] | None:
        """Pick a currently-hot (mcc, hour_bucket) cell, or None to place fraud generically.

        This is the part of the fraud process that *moves*. Cells rotate every few months, so
        knowing which merchant categories are hot right now is knowledge with a short shelf
        life: it can only come from recent labels. A model whose recent labels have been
        censored away keeps declining last quarter's hot categories and misses this quarter's.
        """
        cfg, rng = self.cfg, self.rng
        if rng.uniform() >= cfg.hot_cell_share:
            return None
        cells = ent["hot_cells"][min(month, cfg.n_months - 1)]
        cell = int(cells[rng.integers(0, len(cells))])
        return cell // cfg.n_hour_buckets, cell % cfg.n_hour_buckets

    def _place_in_cell(self, ts: float, bucket: int) -> float:
        """Shift a timestamp into the hour bucket of a hot cell, keeping its calendar day."""
        cfg, rng = self.cfg, self.rng
        width = 24.0 / cfg.n_hour_buckets
        hour = bucket * width + rng.uniform(0.0, width)
        return (ts // SECONDS_PER_DAY) * SECONDS_PER_DAY + hour * 3600.0

    def _merchant_in_mcc(self, mcc: int, ent: dict) -> int:
        members = ent["mcc_members"][mcc]
        if members.size == 0:  # pragma: no cover - only if an mcc drew no merchants
            return int(self.rng.integers(0, self.cfg.n_merchants))
        return int(members[self.rng.integers(0, members.size)])

    def _episode(
        self,
        kind: str,
        card: int,
        t0: float,
        ent: dict[str, np.ndarray],
        breach_merchants: np.ndarray,
        breach_start: np.ndarray,
        breach_len: np.ndarray,
        hot: tuple[int, int] | None = None,
    ) -> list[dict]:
        cfg, rng = self.cfg, self.rng
        mu, sigma = ent["card_mu"][card], ent["card_sigma"][card]
        out: list[dict] = []
        if hot is not None:
            t0 = self._place_in_cell(t0, hot[1])

        if kind == "card_testing":
            # Many tiny authorisations across unrelated merchants inside an hour or so.
            k = int(rng.integers(4, 15))
            gaps = rng.exponential(rng.uniform(40.0, 420.0), size=k).cumsum()
            if hot is not None:
                merch = np.array([self._merchant_in_mcc(hot[0], ent) for _ in range(k)])
            else:
                merch = rng.choice(cfg.n_merchants, size=k, replace=False, p=ent["merchant_pop"])
            dev = int(rng.integers(0, cfg.n_devices))
            amts = np.clip(np.round(rng.uniform(0.8, 22.0, size=k), 2), 0.5, None)
            for j in range(k):
                out.append(self._row(t0 + gaps[j], card, int(merch[j]), dev, ent, amts[j], kind))

        elif kind == "ato_large":
            # 1-3 large-ticket transactions at 02:00-05:00 from a device the card has not used.
            k = int(rng.integers(1, 4))
            night = (
                t0 if hot is not None
                else (t0 // SECONDS_PER_DAY) * SECONDS_PER_DAY + rng.uniform(2, 5) * 3600.0
            )
            dev = int(rng.integers(0, cfg.n_devices))
            merch = (
                np.array([self._merchant_in_mcc(hot[0], ent) for _ in range(k)])
                if hot is not None
                else rng.choice(cfg.n_merchants, size=k, p=ent["merchant_pop"])
            )
            mult = rng.uniform(5.0, 20.0, size=k)
            for j in range(k):
                amt = float(np.clip(np.round(np.exp(mu) * mult[j], 2), 1.0, 32_000.0))
                out.append(
                    self._row(night + j * rng.uniform(120, 2400), card, int(merch[j]), dev, ent, amt, kind)
                )

        elif kind == "merchant_breach":
            # Cards leaked by a compromised merchant get used *at that merchant* in its window.
            b = int(rng.integers(0, len(breach_merchants)))
            t = breach_start[b] + rng.uniform(0.0, breach_len[b])
            k = int(rng.integers(1, 4))
            dev = int(rng.integers(0, cfg.n_devices))
            for j in range(k):
                amt = float(np.clip(np.round(np.exp(rng.normal(mu + 0.8, sigma)), 2), 1.0, 32_000.0))
                out.append(
                    self._row(
                        t + j * rng.uniform(600, 86_400), card, int(breach_merchants[b]), dev, ent, amt, kind
                    )
                )

        else:  # stealth_small — deliberately hard: in-distribution amount, normal hour, own device
            k = int(rng.integers(1, 3))
            dev = (
                int(ent["card_device"][card])
                if rng.uniform() < 0.55
                else int(rng.integers(0, cfg.n_devices))
            )
            b = ent["baskets"][card]
            for j in range(k):
                if hot is not None:
                    merch = self._merchant_in_mcc(hot[0], ent)
                else:
                    merch = (
                        int(b[rng.integers(0, len(b))])
                        if rng.uniform() < 0.45
                        else int(rng.choice(cfg.n_merchants, p=ent["merchant_pop"]))
                    )
                amt = float(np.clip(np.round(np.exp(rng.normal(mu + 0.25, sigma)), 2), 1.0, 32_000.0))
                t = t0 + j * rng.uniform(3600, 172_800)
                if hot is not None:
                    t = self._place_in_cell(t, hot[1])
                out.append(self._row(t, card, merch, dev, ent, amt, kind))

        return out

    def _row(
        self,
        ts: float,
        card: int,
        merchant: int,
        device: int,
        ent: dict[str, np.ndarray],
        amount: float,
        kind: str,
    ) -> dict:
        return {
            "ts": float(ts) + SECONDS_PER_DAY,
            "card_ix": int(card),
            "merchant_ix": int(merchant),
            "device_ix": int(device),
            "addr_ix": int(ent["card_addr"][card]),
            "email_ix": int(ent["card_email"][card]),
            "amount": float(amount),
            "archetype": kind,
        }

    # ------------------------------------------------------------------ latent V columns
    def _v_columns(self, df: pd.DataFrame, ent: dict[str, np.ndarray]) -> pd.DataFrame:
        """Noisy projections of the latent risk state, mimicking IEEE-CIS's engineered `V*`.

        These are *current-transaction* attributes: a bounded, noisy read on how anomalous this
        authorisation looks to the issuer's upstream scoring stack. They carry real but partial
        signal, so a model that also has velocity features beats one that does not.
        """
        cfg, rng = self.cfg, self.rng
        n = len(df)
        merch_risk = ent["merchant_risk"][df["merchant_ix"].to_numpy()]
        own_device = (df["device_ix"].to_numpy() == ent["card_device"][df["card_ix"].to_numpy()])
        hour = ((df["ts"].to_numpy() % SECONDS_PER_DAY) / 3600.0)
        amt_z = (
            np.log(df["amount"].to_numpy()) - ent["card_mu"][df["card_ix"].to_numpy()]
        ) / np.maximum(ent["card_sigma"][df["card_ix"].to_numpy()], 1e-3)

        latent = np.column_stack(
            [
                merch_risk,
                (~own_device).astype(float),
                np.exp(-0.5 * ((hour - 3.5) / 2.0) ** 2),
                np.clip(amt_z, -4, 6),
                df["is_fraud"].to_numpy(dtype=float),  # partially observable "fraud smell"
            ]
        )
        n_signal = cfg.n_v_columns - cfg.n_v_noise
        # Random loadings; the label column gets a deliberately small, noisy loading so the V
        # block alone cannot solve the task (mirrors IEEE-CIS, where V* help but do not suffice).
        loadings = rng.normal(0.0, 1.0, size=(latent.shape[1], n_signal))
        loadings[4, :] *= 0.18
        signal = latent @ loadings
        signal = (signal - signal.mean(0)) / (signal.std(0) + 1e-9)
        signal = signal + rng.normal(0.0, 1.6, size=signal.shape)
        noise = rng.normal(0.0, 1.0, size=(n, cfg.n_v_noise))
        block = np.round(np.column_stack([signal, noise]), 4)
        cols = {f"v{i + 1}": block[:, i] for i in range(cfg.n_v_columns)}
        # IEEE-CIS `V*` are famously sparse; blank out a realistic fraction.
        for i, c in enumerate(cols):
            if i % 4 == 3:
                miss = rng.uniform(size=n) < 0.22
                cols[c] = np.where(miss, np.nan, cols[c])
        return pd.DataFrame(cols, index=df.index)

    # ------------------------------------------------------------------ entry point
    def generate(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n_fraud = int(round(cfg.n_transactions * cfg.fraud_rate))
        n_legit = cfg.n_transactions - n_fraud
        log.info("generating %d legit + %d fraud transactions", n_legit, n_fraud)

        ent = self._build_entities()
        legit = self._legit_frame(n_legit, ent)
        fraud = self._fraud_frame(n_fraud, ent)

        df = pd.concat([legit, fraud], ignore_index=True)
        df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
        df.insert(0, "transaction_id", np.arange(1, len(df) + 1, dtype=np.int64))

        df["product_cd"] = rng.choice(
            list(cfg.product_codes), size=len(df), p=np.asarray(cfg.product_weights)
        )
        v = self._v_columns(df, ent)

        out = pd.DataFrame(
            {
                "transaction_id": df["transaction_id"],
                "ts": df["ts"].astype("float64"),
                "amount": df["amount"].astype("float64"),
                "card_id": ("c" + df["card_ix"].astype(str)).astype("string"),
                "merchant_id": ("m" + df["merchant_ix"].astype(str)).astype("string"),
                "device_id": ("d" + df["device_ix"].astype(str)).astype("string"),
                "email_domain": pd.Series(
                    [cfg.email_domains[i] for i in df["email_ix"]], index=df.index
                ).astype("string"),
                "addr_id": ("a" + df["addr_ix"].astype(str)).astype("string"),
                "product_cd": df["product_cd"].astype("string"),
                "is_fraud": df["is_fraud"].astype("int8"),
                # Merchant category code: an ordinary categorical the model sees. Combined with
                # hour of day it spans the cell grid that fraud rotates through.
                "mcc": ent["merchant_mcc"][df["merchant_ix"].to_numpy()].astype("int16"),
                # Kept for analysis and *never* exposed to any model (see FeatureBuilder).
                "archetype": df["archetype"].astype("string"),
            }
        )
        out = pd.concat([out, v], axis=1)
        return validate_frame(out)


def generate_stream(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper: draw a full labelled stream."""
    return SyntheticStreamGenerator(config).generate()
