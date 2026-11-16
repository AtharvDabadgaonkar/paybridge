# PayBridge — Unit Economics

Every figure here is a **documented modelling assumption** for a synthetic simulation. None of it is
a provider rate card, and none of the savings are realised savings. All assumptions live in
`src/paybridge/config.py`; change them there and every number in the product moves with them.

---

## 1. Cost model

**Customer price** (what the business pays PayBridge):

```
fixed_fee        = rail.customer_fixed_fee_usd
variable_fee     = amount_usd × rail.customer_variable_pct
fx_markup_cost   = amount_usd × rail.customer_fx_markup_bps / 10,000
lifting_fees     = SWIFT only: corridor.swift_hops × $15.00  (pass-through)
───────────────────────────────────────────────────────────────────────
total_customer_cost = fixed_fee + variable_fee + fx_markup_cost + lifting_fees
```

**Platform direct cost** (what PayBridge pays providers):

```
provider_fixed_cost    = rail.provider_fixed_cost_usd
provider_variable_cost = amount_usd × rail.provider_variable_pct
partner_payout_cost    = corridor.partner_payout_fee_usd × rail partner factor
fx_spread_cost         = amount_usd × corridor.fx_wholesale_spread_bps / 10,000
lifting_fees           = same pass-through amount as above
───────────────────────────────────────────────────────────────────────
platform_direct_cost = sum of the above
contribution_margin  = total_customer_cost − platform_direct_cost
```

**FX.** `offered_rate = mid_market_rate × (1 − customer_fx_markup_bps / 10,000)`, and the recipient
receives `(amount − fixed_fee − variable_fee − lifting_fees) × offered_rate`. Mid-market rates are
indicative synthetic constants in `config.FX_MID_RATES`.

## 2. Rail assumptions

| Parameter | ACH funding + partner payout | SWIFT wire | USDC settlement |
| --- | --- | --- | --- |
| Customer fixed fee | $2.50 | $25.00 | $1.00 |
| Customer variable | 0.35% | 0.10% | 0.60% |
| Customer FX markup | 90 bps | 250 bps | 50 bps |
| Correspondent fees | — | $15.00 per hop | — |
| Provider fixed cost | $0.80 | $12.00 | $0.50 |
| Provider variable cost | 0.08% | 0.02% | 0.25% |
| Local partner cost factor | 1.0 | 0.0 | 0.7 |
| Settlement p50 / p90 | 28 h / 62 h | 52 h / 108 h | 1.5 h / 7 h |
| Amount limits | $1 – $250,000 | $100 – $5,000,000 | $10 – $500,000 |
| Expected success rate | 97.5% | 96.2% | 98.1% |

**ACH is modelled correctly as a domestic US network.** It covers the USD funding leg only; the
destination leg is a local banking partner. ACH is never modelled as a standalone international
settlement network.

**USDC is simulated end to end.** On-ramp, transfer and local off-ramp costs are modelled. No wallet
is created, no key is held, no chain or exchange is contacted, and no cryptocurrency moves.

## 3. Corridor assumptions

| Corridor | Currency | Volume share | Partner payout fee | Wholesale FX spread | Risk weight | SWIFT hops |
| --- | --- | --- | --- | --- | --- | --- |
| US-MX | MXN | 22% | $1.40 | 18 bps | 0.35 | 1 |
| US-IN | INR | 19% | $1.10 | 15 bps | 0.30 | 1 |
| US-PH | PHP | 14% | $1.75 | 22 bps | 0.45 | 2 |
| US-GB | GBP | 13% | $0.90 | 10 bps | 0.15 | 1 |
| US-NG | NGN | 9% | $2.60 | 45 bps | 0.75 | 2 |
| US-BR | BRL | 9% | $1.95 | 28 bps | 0.50 | 2 |
| US-DE | EUR | 8% | $0.85 | 9 bps | 0.12 | 1 |
| US-SG | SGD | 6% | $1.05 | 12 bps | 0.18 | 1 |

## 4. Worked example — $5,000 to Mexico

| | ACH + partner | SWIFT | USDC |
| --- | --- | --- | --- |
| Fixed fee | $2.50 | $25.00 | $1.00 |
| Variable fee | $17.50 | $5.00 | $30.00 |
| FX markup | $45.00 | $125.00 | $25.00 |
| Correspondent fees | — | $15.00 | — |
| **Customer pays** | **$65.00** | **$170.00** | **$56.00** |
| Effective cost | 1.30% | 3.40% | 1.12% |
| Platform direct cost | $15.20 | $37.00 | $22.98 |
| **Contribution margin** | **$49.80** | **$133.00** | **$33.02** |
| Settlement p50 | ~31 h | ~58 h | ~1.6 h |

## 5. The 60% hypothesis — tested, not assumed

**Hypothesis:** alternative rails can deliver roughly 60% lower modelled payout cost than SWIFT for
specific scenarios.

**Result (calculated, US-MX):**

| Amount | USDC vs SWIFT | ACH + partner vs SWIFT |
| --- | --- | --- |
| $500 | **87.7%** | **83.5%** |
| $5,000 | **67.1%** | **61.8%** |
| $50,000 | **58.9%** | **53.2%** |

**Portfolio-wide, the realised synthetic rail mix saves 46.1%** against an all-SWIFT counterfactual
($2.20M on $140.5M of volume) — lower than 60% because the mix still contains 23% SWIFT payouts,
which save nothing by construction.

**Interpretation.** The hypothesis holds for small and mid-size payouts and breaks down as amounts
grow. SWIFT's cost is dominated by fixed wire and correspondent fees, which amortise away on large
transfers, while percentage-based FX and processing costs do not. The product implication is that a
rail recommendation is most valuable below roughly $50,000, and that above it the pitch has to be
settlement speed and predictability rather than cost. The Payments page ships a corridor × amount
sensitivity grid so this can be checked for any scenario, including the ones where the answer is
unfavourable.

## 6. Risk economics

| Assumption | Value | Used for |
| --- | --- | --- |
| Loss given undetected fraud | 85% of the payout amount | Fraud-loss simulation |
| Manual review cost | $18.00 per case | Review-cost comparison |

```
prevented_loss = Σ amount(caught fraud)  × loss_rate
incurred_loss  = Σ amount(missed fraud)  × loss_rate
review_cost    = alerts × review_cost_per_case
total_cost     = incurred_loss + review_cost
net_benefit    = (Σ amount(all fraud) × loss_rate) − total_cost
```

Both assumptions are adjustable in the Model Evaluation page so the sensitivity of any conclusion
to them is visible rather than hidden.

## 7. Platform economics (synthetic portfolio)

| Metric | Value |
| --- | --- |
| Payout volume | $140.5M |
| Transactions | 24,961 |
| Revenue (customer fees) | $2.57M |
| Platform direct cost | $0.63M |
| Contribution margin | $1.94M (75.4% of revenue) |
| Average revenue per transaction | $102.92 |
| Average direct cost per payout | $25.27 |
| Average contribution per payout | $77.65 |

A 75% contribution margin is a *gross* margin on direct payment costs only. It excludes engineering,
compliance, support, capital and risk-loss provisioning, so it is not a profit margin and should not
be read as one.

## 8. Sensitivity and limitations

- Fees, FX markups, partner costs and spreads are indicative. Real pricing is negotiated per
  provider, per corridor and per volume tier.
- Correspondent lifting fees vary widely by bank and route; a single $15/hop assumption is a
  simplification.
- No float income, treasury carry, FX hedging cost, chargeback cost, or failed-payment repair cost
  is modelled.
- Success rates and settlement distributions are simulated, not observed.
- The counterfactual assumes a business *could* have used SWIFT for every payout, which overstates
  the comparison where SWIFT would have been impractical and understates it where it was the only
  real option.
