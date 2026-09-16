from datetime import date
from .domain import OptionContract
from .analysis import option_liquidity_score


def contracts_from_chain(chain, underlying: str, expiration: date, option_type: str = "call") -> list[OptionContract]:
    table = chain.calls if option_type == "call" else chain.puts
    contracts: list[OptionContract] = []
    for row in table.to_dict("records"):
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        volume = int(row.get("volume") or 0)
        oi = int(row.get("openInterest") or 0)
        if bid < 0 or ask <= 0 or ask < bid:
            continue
        contracts.append(OptionContract(
            symbol=str(row.get("contractSymbol", "")), underlying=underlying,
            strike=float(row["strike"]), expiration=expiration, option_type=option_type,
            bid=bid, ask=ask, volume=volume, open_interest=oi,
            implied_volatility=float(row.get("impliedVolatility")) if row.get("impliedVolatility") else None,
            delta=None, gamma=None, theta=None, vega=None,
        ))
    return contracts


def filter_contracts(contracts: list[OptionContract], min_dte: int, max_dte: int, max_spread_pct: float, min_volume: int, min_oi: int):
    return [c for c in contracts if min_dte <= c.dte <= max_dte and c.spread_percentage <= max_spread_pct and c.volume >= min_volume and c.open_interest >= min_oi and option_liquidity_score(c.bid, c.ask, c.volume, c.open_interest) > 0]
