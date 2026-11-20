# GTA 6 Trophy Rarity Forecast

A deterministic, standard-library-only Python program that forecasts GTA VI PS5 trophy earn rates.

It reads a case file from `/input/<id>.in` and writes five quantiles (5/25/50/75/95%) of every trophy's earn rate at the case target time to `/output/<id>.out`, fitting saturating growth curves per trophy on the logit scale.

Entry for the make.fun bounty `gta6-trophy-rarity-forecast-01`. MIT licensed.
