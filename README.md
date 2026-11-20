[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/OG2BIT001/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/OG2BIT001/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA 6 Trophy Rarity Forecast

A deterministic, standard-library-only Python program that forecasts GTA VI PS5 trophy earn rates.

It reads a case file from `/input/<id>.in` and writes five quantiles (5/25/50/75/95%) of every trophy's earn rate at the case target time to `/output/<id>.out`. Per trophy it fits saturating growth curves on the logit scale (a log-time linear fit competing with an exponential approach to a fitted asymptote), falls back to persistence for sparse series, and derives its quantile spread from fit residuals, prediction leverage, the sampling noise of the last read and a horizon random-walk term.

Run it:

```
python3 main.py
```

Entry for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01). MIT licensed.
